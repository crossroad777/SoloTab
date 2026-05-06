"""
train_guitarset_ft_all.py — 6ドメインモデルのGuitarSet直接ファインチューニング
==============================================================================
合成データ専門モデルを実録音(GuitarSet)で追加訓練し、
Pure MoEアンサンブルの実音源検出能力を底上げする。

Usage:
    python train_guitarset_ft_all.py
    python train_guitarset_ft_all.py --domain martin_finger
"""
import os, sys, io, json, time, argparse
import torch
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
from data_processing import preparation, dataset
from data_processing.dataset import create_frame_level_labels
from training import loss_functions, epoch_processing
from model import architecture
import config

DOMAINS = [
    "martin_finger", "taylor_finger", "luthier_finger",
    "martin_pick", "taylor_pick", "luthier_pick",
    "gibson_thumb",
]

OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")
TRAINING_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output")

FT_LR = 5e-6
FT_EPOCHS_DEFAULT = 9999
FT_PATIENCE_DEFAULT = 10
FT_BATCH_SIZE = 2


def ensure_guitarset_preprocessed():
    """GuitarSetの前処理済みデータが存在するか確認し、なければ作成"""
    ids_path = os.path.join(OUTPUT_BASE_DIR, "train_ids.txt")
    if os.path.exists(ids_path):
        return
    print("[!] Preprocessed GuitarSet data not found. Running preparation...")
    track_splits = preparation.prepare_track_splits(
        data_home=config.DATA_HOME_DEFAULT,
        problematic_files_list=config.PROBLEMATIC_FILES,
        test_split_fraction=config.TEST_SPLIT_SIZE,
        validation_split_fraction=config.VALIDATION_SPLIT_SIZE,
        seed=config.RANDOM_SEED,
        output_dir_for_ids=OUTPUT_BASE_DIR,
    )
    preparation.preprocess_guitarset_data(
        guitarset_data_home=config.DATA_HOME_DEFAULT,
        processed_output_base_dir=OUTPUT_BASE_DIR,
        track_ids_map=track_splits,
        audio_sample_rate=config.SAMPLE_RATE,
        audio_hop_length=config.HOP_LENGTH,
        audio_n_cqt_bins=config.N_BINS_CQT,
        audio_cqt_bins_per_octave=config.BINS_PER_OCTAVE_CQT,
        audio_cqt_fmin=config.FMIN_CQT,
    )


def parse_log(log_path):
    """ログから最終Epoch番号とBest F1を取得"""
    start_epoch = 1
    best_f1 = 0.0
    already_early_stopped = False
    if not os.path.exists(log_path):
        return start_epoch, best_f1, already_early_stopped
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        found_epoch = False
        for line in reversed(lines):
            s = line.strip()
            if "Best F1:" in s:
                val = float(s.split("Best F1:")[1].split()[0])
                best_f1 = max(best_f1, val)
            if not found_epoch and s.startswith("Epoch ") and "|" in s:
                ep = int(s.split("|")[0].strip().split()[1])
                start_epoch = ep + 1
                found_epoch = True
        # Check for early stop (forward scan)
        for line in lines:
            if "Early stop at epoch" in line:
                already_early_stopped = True
                break
    except Exception:
        pass
    return start_epoch, best_f1, already_early_stopped


def finetune_one(domain, device, ft_epochs=FT_EPOCHS_DEFAULT, ft_patience=FT_PATIENCE_DEFAULT):
    print(f"\n{'='*60}")
    print(f"  GuitarSet FT: {domain}")
    print(f"{'='*60}")

    # pre-emajorモデル(GuitarSet FT済み)をベースに追加FT
    pre_emajor = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_guitarset_ft", "best_model_pre_emajor.pth")
    domain_src = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_model", "best_model.pth")
    src_path = pre_emajor if os.path.exists(pre_emajor) else domain_src
    out_dir = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_guitarset_ft")
    cfg_path = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_model", "run_configuration.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(TRAINING_OUTPUT_DIR, "baseline_model", "run_configuration.json")

    if not os.path.exists(src_path):
        print(f"[SKIP] Source not found: {src_path}")
        return

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pth")
    log_path = os.path.join(out_dir, "training_log.txt")

    start_epoch, best_f1, already_early_stopped = parse_log(log_path)
    if already_early_stopped:
        print(f"[SKIP] Already early-stopped for {domain} (Best F1={best_f1:.4f})")
        return
    if start_epoch > ft_epochs:
        print(f"[SKIP] Already completed for {domain} (epoch {start_epoch-1}/{ft_epochs})")
        return

    load_path = best_path if (start_epoch > 1 and os.path.exists(best_path)) else src_path
    print(f"  Loading from: {load_path} (epoch {start_epoch})")

    with open(cfg_path, "r", encoding="utf-8") as f:
        hp = json.load(f)["hyperparameters_tuned"]

    # Data loaders (aug=Off: 前処理済みCQT特徴量を使用)
    common = config.DATASET_COMMON_PARAMS
    train_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name="train",
        guitarset_data_home=config.DATA_HOME_DEFAULT,
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_EVAL_AUGMENTATION_PARAMS,
    )
    
    # E major合成データを追加（前処理済みCQTを直接読み込み）
    emajor_ids_path = os.path.join(OUTPUT_BASE_DIR, "emajor_train_ids.txt")
    if os.path.exists(emajor_ids_path):
        from torch.utils.data import ConcatDataset, Dataset as TorchDataset
        
        class SynthDataset(TorchDataset):
            """前処理済み合成データの簡易Dataset"""
            def __init__(self, base_dir, ids_file, hop_length, sr, max_fret):
                self.items = []
                self.hop_length = hop_length
                self.sr = sr
                self.max_fret = max_fret
                with open(ids_file, 'r') as f:
                    for line in f:
                        tid = line.strip()
                        if not tid:
                            continue
                        feat_path = os.path.join(base_dir, f"{tid}_features.pt")
                        label_path = os.path.join(base_dir, f"{tid}_labels.pt")
                        if os.path.exists(feat_path) and os.path.exists(label_path):
                            self.items.append((feat_path, label_path, tid))
            
            def __len__(self):
                return len(self.items)
            
            def __getitem__(self, idx):
                feat_path, label_path, tid = self.items[idx]
                features = torch.load(feat_path, weights_only=False)
                raw_labels = torch.load(label_path, weights_only=False)
                labels_tuple = create_frame_level_labels(
                    raw_labels, features, self.hop_length, self.sr, self.max_fret
                )
                return features, labels_tuple, raw_labels, tid
        
        train_data_dir = os.path.join(OUTPUT_BASE_DIR, "train")
        emajor_ds = SynthDataset(
            train_data_dir, emajor_ids_path,
            hop_length=config.HOP_LENGTH,
            sr=config.SAMPLE_RATE,
            max_fret=common.get('max_fret_value', 19),
        )
        if len(emajor_ds) > 0:
            combined_ds = ConcatDataset([train_ds, emajor_ds])
            print(f"  GuitarSet: {len(train_ds)}, E major synth: {len(emajor_ds)}")
            train_ds = combined_ds
    
    val_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name="validation",
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_EVAL_AUGMENTATION_PARAMS,
    )
    train_loader = DataLoader(train_ds, batch_size=FT_BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_pad, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=FT_BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_pad, num_workers=0)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
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
    sd = torch.load(load_path, map_location=device, weights_only=False)
    if list(sd.keys())[0].startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)

    onset_pw = (torch.tensor([hp["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device)
                if hp.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None)
    criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pw, onset_loss_weight=hp["ONSET_LOSS_WEIGHT"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=hp["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=7)

    # Training loop
    bad_epochs = 0
    mode = "a" if start_epoch > 1 else "w"
    with open(log_path, mode, encoding="utf-8") as log_f:
        if mode == "w":
            log_f.write(f"--- GuitarSet FT for {domain} | {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        for epoch in range(start_epoch, ft_epochs + 1):
            desc = f"Epoch {epoch}/{ft_epochs}"

            train_pbar = tqdm(train_loader, desc=f"{desc} [Train]", unit="b", leave=False, dynamic_ncols=True)
            train_m = epoch_processing.train_one_epoch(model, train_pbar, optimizer, criterion, device)

            val_pbar = tqdm(val_loader, desc=f"{desc} [Val]", unit="b", leave=False, dynamic_ncols=True)
            val_m = epoch_processing.evaluate_one_epoch(model, val_pbar, criterion, device, config)

            f1 = val_m.get("val_tdr_f1_at_0.5", 0.0)
            p = val_m.get("val_tdr_precision_at_0.5", 0.0)
            r = val_m.get("val_tdr_recall_at_0.5", 0.0)
            scheduler.step(f1)

            line = f"Epoch {epoch} | Loss: {train_m['train_total_loss']:.4f} | F1: {f1:.4f} P: {p:.4f} R: {r:.4f}"
            print(line)
            log_f.write(line + "\n")

            if f1 >= best_f1:
                best_f1 = f1
                bad_epochs = 0
                torch.save(model.state_dict(), best_path)
                msg = f"  -> Best F1: {best_f1:.4f} saved!"
                print(msg)
                log_f.write(msg + "\n")
            else:
                bad_epochs += 1
                if bad_epochs >= ft_patience:
                    print(f"  Early stop at epoch {epoch}")
                    log_f.write(f"  Early stop at epoch {epoch}\n")
                    break
            log_f.flush()

    # Cleanup
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    print(f"  Completed {domain}: Best F1 = {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None, help="Single domain to fine-tune")
    parser.add_argument("--epochs", type=int, default=FT_EPOCHS_DEFAULT, help=f"Max epochs (default: {FT_EPOCHS_DEFAULT})")
    parser.add_argument("--patience", type=int, default=FT_PATIENCE_DEFAULT, help=f"Early stopping patience (default: {FT_PATIENCE_DEFAULT})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ensure_guitarset_preprocessed()

    domains = [args.domain] if args.domain else DOMAINS
    for d in domains:
        finetune_one(d, device, ft_epochs=args.epochs, ft_patience=args.patience)

    print("\n" + "=" * 60)
    print("All GuitarSet fine-tuning complete!")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
