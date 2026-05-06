"""
train_distill.py — 6ドメインFTモデル → 単一モデル蒸留
=====================================================
GuitarSet-FT済みの6ドメインモデル(教師)の知識を
単一のGuitarTabCRNN(生徒)に蒸留する。

推論時に6モデル×78MBをロードする必要がなくなり、
1モデル(78MB)で同等以上の性能を目指す。

Usage:
    python train_distill.py
"""
import os, sys, io, json, time
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ["TQDM_ASCII"] = " 123456789#"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from data_processing import dataset
from data_processing.dataset import create_frame_level_labels
from training import loss_functions, epoch_processing
from model import architecture
import config

DOMAINS = [
    "martin_finger", "taylor_finger", "luthier_finger",
    "martin_pick", "taylor_pick", "luthier_pick",
]

OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")
TRAINING_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output")
RUN_DIR = os.path.join(TRAINING_OUTPUT_DIR, "distilled_unified_model")

DISTILL_LR = 3e-5
DISTILL_EPOCHS = 9999
DISTILL_PATIENCE = 10
DISTILL_BATCH_SIZE = 2
TEMPERATURE = 3.0
ALPHA = 0.7  # 蒸留Loss の重み (1-ALPHA = GT Loss の重み)


def load_teacher_models(device):
    """6つのGuitarSet-FT済み教師モデルをロード"""
    teachers = []
    cfg_path = os.path.join(TRAINING_OUTPUT_DIR, "baseline_model", "run_configuration.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        hp = json.load(f)["hyperparameters_tuned"]

    with torch.no_grad():
        cnn_out_dim = architecture.TabCNN()(torch.randn(1, 1, config.N_BINS_CQT, 32))
        cnn_out_dim = cnn_out_dim.shape[1] * cnn_out_dim.shape[2]

    for domain in DOMAINS:
        # GuitarSet FT版を優先、なければ元のドメインモデル
        ft_path = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_guitarset_ft", "best_model.pth")
        orig_path = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_model", "best_model.pth")
        model_path = ft_path if os.path.exists(ft_path) else orig_path

        if not os.path.exists(model_path):
            print(f"[WARN] Teacher not found: {domain}, skipping")
            continue

        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=cnn_out_dim,
            rnn_type=hp.get("RNN_TYPE", "GRU"),
            rnn_hidden_size=hp["RNN_HIDDEN_SIZE"],
            rnn_layers=hp["RNN_LAYERS"],
            rnn_dropout=hp["RNN_DROPOUT"],
            rnn_bidirectional=hp.get("RNN_BIDIRECTIONAL", True),
        )
        sd = torch.load(model_path, map_location=device, weights_only=False)
        if list(sd.keys())[0].startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append((domain, model))
        src = "GS-FT" if os.path.exists(ft_path) else "Synth"
        print(f"  Teacher loaded: {domain} ({src})")

    print(f"  Total teachers: {len(teachers)}")
    return teachers, hp, cnn_out_dim


def get_teacher_outputs(teachers, features, device):
    """教師アンサンブルのソフト出力を取得"""
    all_onset_logits = []
    all_fret_logits = []

    with torch.no_grad():
        for _, teacher in teachers:
            onset_logits, fret_logits = teacher(features)
            all_onset_logits.append(onset_logits)
            all_fret_logits.append(fret_logits)

    # onset: 平均 logits (soft ensemble)
    stacked_onset = torch.stack(all_onset_logits, dim=0)  # [N_teachers, B, T, 6]
    avg_onset_logits = stacked_onset.mean(dim=0)           # [B, T, 6]

    # fret: 平均 logits → soft target
    stacked_fret = torch.stack(all_fret_logits, dim=0)     # [N_teachers, B, T, 6, C]
    avg_fret_logits = stacked_fret.mean(dim=0)             # [B, T, 6, C]

    return avg_onset_logits, avg_fret_logits


def distill_loss(student_onset, student_fret, teacher_onset, teacher_fret, T):
    """蒸留Loss: KL divergence on softened outputs"""
    # Onset: binary → sigmoid + MSE on soft targets
    teacher_soft = torch.sigmoid(teacher_onset / T)
    student_soft = torch.sigmoid(student_onset / T)
    onset_distill = F.mse_loss(student_soft, teacher_soft) * (T * T)

    # Fret: KL divergence on softened class probs
    B, Tf, S, C = student_fret.shape
    student_log_soft = F.log_softmax(student_fret.view(-1, C) / T, dim=-1)
    teacher_soft_fret = F.softmax(teacher_fret.view(-1, C) / T, dim=-1)
    fret_distill = F.kl_div(student_log_soft, teacher_soft_fret, reduction='batchmean') * (T * T)

    return onset_distill + fret_distill


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("  Knowledge Distillation: 6 Teachers → 1 Student")
    print("=" * 60)

    os.makedirs(RUN_DIR, exist_ok=True)
    best_path = os.path.join(RUN_DIR, "best_model.pth")
    log_path = os.path.join(RUN_DIR, "training_log.txt")

    # Resume logic
    start_epoch = 1
    best_f1 = 0.0
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    s = line.strip()
                    if "Best F1:" in s:
                        best_f1 = max(best_f1, float(s.split("Best F1:")[1].split()[0]))
                    if s.startswith("Epoch ") and "|" in s:
                        start_epoch = int(s.split("|")[0].strip().split()[1]) + 1
                        break
        except Exception:
            pass

    if start_epoch > DISTILL_EPOCHS:
        print("[DONE] Distillation already completed.")
        return

    # Load teachers
    print("\n[1] Loading teacher models...")
    teachers, hp, cnn_out_dim = load_teacher_models(device)
    if len(teachers) < 3:
        print("[ERROR] Need at least 3 teacher models.")
        return

    # Data loaders
    print("\n[2] Preparing data loaders...")
    common = config.DATASET_COMMON_PARAMS
    train_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name="train",
        guitarset_data_home=config.DATA_HOME_DEFAULT,
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_TRAIN_AUGMENTATION_PARAMS,
    )
    val_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name="validation",
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_EVAL_AUGMENTATION_PARAMS,
    )
    train_loader = DataLoader(train_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_pad, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=DISTILL_BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_pad, num_workers=0)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Student model (from baseline)
    print("\n[3] Creating student model...")
    student = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hp.get("RNN_TYPE", "GRU"),
        rnn_hidden_size=hp["RNN_HIDDEN_SIZE"],
        rnn_layers=hp["RNN_LAYERS"],
        rnn_dropout=hp["RNN_DROPOUT"],
        rnn_bidirectional=hp.get("RNN_BIDIRECTIONAL", True),
    )
    # Initialize from baseline
    baseline_path = os.path.join(TRAINING_OUTPUT_DIR, "baseline_model", "best_model.pth")
    resume_path = best_path if (start_epoch > 1 and os.path.exists(best_path)) else baseline_path
    sd = torch.load(resume_path, map_location=device, weights_only=False)
    if list(sd.keys())[0].startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    student.load_state_dict(sd)
    student.to(device)
    print(f"  Student loaded from: {resume_path}")

    onset_pw = (torch.tensor([hp["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device)
                if hp.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None)
    gt_criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pw, onset_loss_weight=hp["ONSET_LOSS_WEIGHT"]
    ).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=DISTILL_LR, weight_decay=hp["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=7)

    # Training
    print(f"\n[4] Training: epoch {start_epoch}-{DISTILL_EPOCHS}, alpha={ALPHA}, T={TEMPERATURE}")
    bad_epochs = 0
    mode = "a" if start_epoch > 1 else "w"

    with open(log_path, mode, encoding="utf-8") as log_f:
        if mode == "w":
            log_f.write(f"--- Distillation | {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_f.write(f"Teachers: {len(teachers)}, alpha={ALPHA}, T={TEMPERATURE}\n")

        for epoch in range(start_epoch, DISTILL_EPOCHS + 1):
            student.train()
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{DISTILL_EPOCHS} [Train]",
                        unit="b", leave=False, dynamic_ncols=True)
            for batch in pbar:
                features, (onset_gt, fret_gt), lengths, *_ = batch
                features = features.to(device)
                onset_gt = onset_gt.to(device)
                fret_gt = fret_gt.to(device)
                lengths = lengths.to(device)

                # Student forward
                s_onset, s_fret = student(features)

                # Teacher ensemble output
                t_onset, t_fret = get_teacher_outputs(teachers, features, device)

                # Combined loss
                gt_loss, _, _ = gt_criterion(s_onset, s_fret, onset_gt, fret_gt, lengths)
                d_loss = distill_loss(s_onset, s_fret, t_onset, t_fret, TEMPERATURE)
                loss = ALPHA * d_loss + (1 - ALPHA) * gt_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / max(1, n_batches)

            # Validation (standard GT-based evaluation)
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{DISTILL_EPOCHS} [Val]",
                            unit="b", leave=False, dynamic_ncols=True)
            val_m = epoch_processing.evaluate_one_epoch(student, val_pbar, gt_criterion, device, config)

            f1 = val_m.get("val_tdr_f1_at_0.5", 0.0)
            p = val_m.get("val_tdr_precision_at_0.5", 0.0)
            r = val_m.get("val_tdr_recall_at_0.5", 0.0)
            scheduler.step(f1)

            line = f"Epoch {epoch} | Loss: {avg_loss:.4f} | F1: {f1:.4f} P: {p:.4f} R: {r:.4f}"
            print(line)
            log_f.write(line + "\n")

            if f1 >= best_f1:
                best_f1 = f1
                bad_epochs = 0
                torch.save(student.state_dict(), best_path)
                msg = f"  -> Best F1: {best_f1:.4f} saved!"
                print(msg)
                log_f.write(msg + "\n")
            else:
                bad_epochs += 1
                if bad_epochs >= DISTILL_PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    log_f.write(f"  Early stop at epoch {epoch}\n")
                    break
            log_f.flush()

    # Cleanup teachers
    for _, t in teachers:
        del t
    del student
    torch.cuda.empty_cache()

    print(f"\nDistillation complete! Best F1: {best_f1:.4f}")
    print(f"Model saved: {best_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
