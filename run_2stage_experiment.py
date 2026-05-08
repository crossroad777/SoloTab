"""
run_2stage_experiment.py — 2段階学習テスト
============================================
Stage 1: Synth V2 (5000件) で事前学習
Stage 2: 実データ (GuitarSet+GAPS+AG-PT) でFT

Usage:
    # 小テスト（Stage 1: 3エポック → Stage 2: 5エポック）
    python run_2stage_experiment.py

    # フル実行
    python run_2stage_experiment.py --s1-epochs 15 --s2-epochs 100 --s2-patience 15
"""
import os, sys, io, shutil, json, time, argparse, re
os.chdir(r"d:\Music\nextchord-solotab\music-transcription\python")
sys.path.insert(0, r"d:\Music\nextchord-solotab")
sys.path.insert(0, r"d:\Music\nextchord-solotab\music-transcription\python")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ["TQDM_ASCII"] = " 123456789#"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

import config
from data_processing.dataset import create_frame_level_labels
from data_processing.batching import collate_fn_pad
from data_processing import dataset as gs_dataset
from model import architecture
from training import loss_functions, epoch_processing

DOMAIN = "martin_finger"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_BASE = r"d:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data"
TRAINING_OUTPUT = os.path.join(OUTPUT_BASE, "training_output")
SYNTH_DIR = r"D:\Music\datasets\synth_v2"
SYNTH_IDS = os.path.join(SYNTH_DIR, "train_ids.txt")

# ソースモデル: 旧ベースライン (F1=0.7927)
GA_DIR = os.path.join(TRAINING_OUTPUT, f"finetuned_{DOMAIN}_multitask_3ds_ga")
BACKUP_MODEL = os.path.join(GA_DIR, "_backup_pre_synth_experiment", "best_model.pth")
EXPERIMENT_DIR = os.path.join(TRAINING_OUTPUT, f"finetuned_{DOMAIN}_synth_v2_experiment")

# HP
domain_src_dir = os.path.join(TRAINING_OUTPUT, f"finetuned_{DOMAIN}_model")
cfg_path = os.path.join(domain_src_dir, "run_configuration.json")
if not os.path.exists(cfg_path):
    cfg_path = os.path.join(TRAINING_OUTPUT, "baseline_model", "run_configuration.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    hp = json.load(f)["hyperparameters_tuned"]


# ============================================================
# Synth Dataset
# ============================================================
class SynthDataset(TorchDataset):
    def __init__(self, base_dir, ids_file):
        self.items = []
        with open(ids_file, 'r') as f:
            for line in f:
                tid = line.strip()
                if not tid: continue
                fp = os.path.join(base_dir, f"{tid}_features.pt")
                lp = os.path.join(base_dir, f"{tid}_labels.pt")
                if os.path.exists(fp) and os.path.exists(lp):
                    self.items.append((fp, lp, tid))
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        fp, lp, tid = self.items[index]
        features = torch.load(fp, weights_only=False)
        raw_labels = torch.load(lp, weights_only=False)
        labels_tuple = create_frame_level_labels(
            raw_labels, features, config.HOP_LENGTH, config.SAMPLE_RATE, config.MAX_FRETS)
        return features, labels_tuple, raw_labels, tid


def build_model():
    with torch.no_grad():
        cnn_out_dim = architecture.TabCNN()(torch.randn(1, 1, config.N_BINS_CQT, 32))
        cnn_out_dim = cnn_out_dim.shape[1] * cnn_out_dim.shape[2]
    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hp.get("RNN_TYPE", "GRU"),
        rnn_hidden_size=hp["RNN_HIDDEN_SIZE"],
        rnn_layers=hp["RNN_LAYERS"],
        rnn_dropout=hp["RNN_DROPOUT"],
        rnn_bidirectional=hp.get("RNN_BIDIRECTIONAL", True),
    )
    return model


def load_weights(model, path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    if list(sd.keys())[0].startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


# ============================================================
# Stage 1: Pre-train on Synth V2
# ============================================================
def run_stage1(model, epochs, lr=1e-4):
    print(f"\n{'='*60}")
    print(f"  Stage 1: Synth V2 Pre-training ({epochs} epochs)")
    print(f"  LR: {lr}")
    print(f"{'='*60}")

    synth_ds = SynthDataset(SYNTH_DIR, SYNTH_IDS)
    print(f"  Synth samples: {len(synth_ds)}")
    synth_loader = DataLoader(synth_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn_pad, num_workers=0, pin_memory=True)

    # GuitarSet val for monitoring
    common = config.DATASET_COMMON_PARAMS
    gs_val = gs_dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE, data_split_name="validation",
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_EVAL_AUGMENTATION_PARAMS,
    )
    val_loader = DataLoader(gs_val, batch_size=1, shuffle=False,
                            collate_fn=collate_fn_pad, num_workers=0, pin_memory=True)

    onset_pw = (torch.tensor([hp["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=DEVICE)
                if hp.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None)
    criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pw, onset_loss_weight=hp["ONSET_LOSS_WEIGHT"]
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=hp["WEIGHT_DECAY"])

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    log_path = os.path.join(EXPERIMENT_DIR, "stage1_log.txt")

    best_f1 = 0.0
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"--- Stage 1: Synth V2 Pre-training | {DOMAIN} | {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_f.write(f"    Synth: {len(synth_ds)}, LR: {lr}\n")

        for epoch in range(1, epochs + 1):
            train_pbar = tqdm(synth_loader, desc=f"S1 E{epoch}/{epochs} [Train]",
                              unit="b", leave=False, dynamic_ncols=True)
            train_m = epoch_processing.train_one_epoch(model, train_pbar, optimizer,
                                                        criterion, DEVICE,
                                                        scaler=None, accumulation_steps=1)

            val_pbar = tqdm(val_loader, desc=f"S1 E{epoch}/{epochs} [Val]",
                            unit="b", leave=False, dynamic_ncols=True)
            val_m = epoch_processing.evaluate_one_epoch(model, val_pbar, criterion,
                                                         DEVICE, config)

            f1 = val_m.get("val_tdr_f1_at_0.5", 0.0)
            p = val_m.get("val_tdr_precision_at_0.5", 0.0)
            r = val_m.get("val_tdr_recall_at_0.5", 0.0)

            line = f"Epoch {epoch} | Loss: {train_m['train_total_loss']:.4f} | F1: {f1:.4f} P: {p:.4f} R: {r:.4f}"
            print(f"  {line}")
            log_f.write(line + "\n")

            if f1 >= best_f1:
                best_f1 = f1
                s1_path = os.path.join(EXPERIMENT_DIR, "stage1_best.pth")
                torch.save(model.state_dict(), s1_path)
                msg = f"  -> Best F1: {best_f1:.4f} saved!"
                print(f"  {msg}")
                log_f.write(msg + "\n")
            log_f.flush()

    print(f"  Stage 1 Complete: Best F1 = {best_f1:.4f}")
    return best_f1


# ============================================================
# Stage 2: FT on real data
# ============================================================
def run_stage2(model, epochs, patience):
    print(f"\n{'='*60}")
    print(f"  Stage 2: Fine-tune on GuitarSet + GAPS + AG-PT")
    print(f"  Epochs: {epochs}, Patience: {patience}")
    print(f"{'='*60}")

    # Stage1のbestモデルをGA_DIRにコピーしてfinetune_multitaskに渡す
    s1_best = os.path.join(EXPERIMENT_DIR, "stage1_best.pth")
    if not os.path.exists(s1_best):
        print("[ERROR] Stage 1 best model not found!")
        return

    # finetune_multitaskが参照するディレクトリ
    s2_dir = os.path.join(TRAINING_OUTPUT, f"finetuned_{DOMAIN}_multitask_3ds")
    os.makedirs(s2_dir, exist_ok=True)
    shutil.copy2(s1_best, os.path.join(s2_dir, "best_model.pth"))

    # ログをクリア
    s2_ga_dir = os.path.join(TRAINING_OUTPUT, f"finetuned_{DOMAIN}_multitask_3ds_ga")
    s2_log = os.path.join(s2_ga_dir, "training_log.txt")
    with open(s2_log, "w", encoding="utf-8") as f:
        f.write(f"--- Stage 2: Real-data FT (from Synth V2) | {DOMAIN} | {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    # Stage1モデルをGA_DIRのbest_model.pthにもコピー（ga_retrain=Trueが参照）
    shutil.copy2(s1_best, os.path.join(s2_ga_dir, "best_model.pth"))

    from backend.train.train_gaps_multitask import finetune_multitask
    finetune_multitask(DOMAIN, DEVICE, ft_epochs=epochs, ft_patience=patience,
                       include_agpt=True, ga_retrain=True)

    # 結果を読み取り
    best_f1 = 0.0
    if os.path.exists(s2_log):
        with open(s2_log, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"Best F1: ([\d.]+)", line)
                if m:
                    best_f1 = max(best_f1, float(m.group(1)))
    print(f"  Stage 2 Complete: Best F1 = {best_f1:.4f}")
    return best_f1


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="2-Stage Training Experiment")
    parser.add_argument("--s1-epochs", type=int, default=3, help="Stage 1 epochs (default: 3)")
    parser.add_argument("--s1-lr", type=float, default=1e-4, help="Stage 1 LR (default: 1e-4)")
    parser.add_argument("--s2-epochs", type=int, default=5, help="Stage 2 epochs (default: 5)")
    parser.add_argument("--s2-patience", type=int, default=5, help="Stage 2 patience (default: 5)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Source model: {BACKUP_MODEL}")

    # Build and load model
    model = build_model()
    load_weights(model, BACKUP_MODEL)
    model.to(DEVICE)
    print(f"Model loaded successfully")

    # Stage 1
    t0 = time.time()
    s1_f1 = run_stage1(model, args.s1_epochs, args.s1_lr)
    s1_time = (time.time() - t0) / 60

    # Stage 2
    t1 = time.time()
    # Reload Stage 1 best for Stage 2
    del model
    torch.cuda.empty_cache()
    model = build_model()
    s1_best = os.path.join(EXPERIMENT_DIR, "stage1_best.pth")
    load_weights(model, s1_best)
    model.to(DEVICE)
    del model  # finetune_multitask will rebuild
    torch.cuda.empty_cache()

    s2_f1 = run_stage2(None, args.s2_epochs, args.s2_patience)
    s2_time = (time.time() - t1) / 60

    # Summary
    print(f"\n{'='*60}")
    print(f"  2-Stage Experiment Summary")
    print(f"{'='*60}")
    print(f"  Baseline (旧GA):  F1 = 0.7927")
    print(f"  Stage 1 (Synth):  F1 = {s1_f1:.4f}  ({s1_time:.1f} min)")
    print(f"  Stage 2 (Real):   F1 = {s2_f1:.4f}  ({s2_time:.1f} min)")
    print(f"  Total:            {(s1_time+s2_time):.1f} min")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
