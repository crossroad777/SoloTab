import os
import sys
import glob
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler, Subset
from torch import optim
from tqdm import tqdm
import torchaudio
import librosa

os.environ["TQDM_ASCII"] = " 123456789#"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from data_processing.preparation import extract_annotations_from_jams
from data_processing.dataset import create_frame_level_labels
from training import loss_functions, epoch_processing
from model import architecture
import config

# Paths
DATASET_BASE_DIR = r"D:\Music\datasets"
OUTALL_DIR = r"D:\Music\all_jams_midi_V2_60000_tracks\outall"  # Synthetic labels
STARTING_MODEL_PATH = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "baseline_model", "best_model.pth")
CONFIG_PATH = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "baseline_model", "run_configuration.json")
OUTPUT_RUN_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "finetuned_multi_dataset")
os.makedirs(OUTPUT_RUN_DIR, exist_ok=True)
NEW_MODEL_PATH = os.path.join(OUTPUT_RUN_DIR, "best_model.pth")
LOG_PATH = os.path.join(OUTPUT_RUN_DIR, "training_log.txt")

# Hyperparameters for Ultimate Fine-Tuning
LR_START = 3e-4         # 4. スクラッチ学習用
EPOCHS = 100
F1_THRESHOLD = 0.00  # 初回から必ず保存させるためハードルをゼロに
PATIENCE_EARLY_STOP = 20  # 自己最高記録を更新できない状態が20回続けば強制終了らストップ

# --- Dataset Class ---
class RobustGuitarDataset(Dataset):
    def __init__(self, tracks_list, augment=True):
        self.tracks = tracks_list
        self.augment = augment
        self.sr = config.SAMPLE_RATE
        self.hop_length = config.HOP_LENGTH
        if self.augment:
            pass # SpecAugment will now be applied on GPU at the batch level

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        cache_path = track["audio"] + ".cqt_multi.pt"
        
        # Load CQT features
        try:
            features = torch.load(cache_path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Cache miss or corrupt: {cache_path}. Run precompute_cqt_caches.py first.")

        # Removed CPU-level SpecAugment for parallel GPU execution

        # Load Labels
        raw_notes = extract_annotations_from_jams(track["jams"])
        if len(raw_notes) > 0:
            raw_labels = torch.tensor(raw_notes, dtype=torch.float32)
        else:
            raw_labels = torch.empty((0, 5))

        onset_labels, fret_labels = create_frame_level_labels(
            raw_labels, features, self.hop_length, self.sr, config.MAX_FRETS
        )

        max_frames = 430
        total_frames = features.shape[-1]
        if total_frames > max_frames:
            start_frame = np.random.randint(0, total_frames - max_frames)
            features = features[:, :, start_frame : start_frame + max_frames]
            onset_labels = onset_labels[start_frame : start_frame + max_frames, :]
            fret_labels = fret_labels[start_frame : start_frame + max_frames, :]
            if len(raw_labels) > 0:
                start_sec = start_frame * self.hop_length / self.sr
                end_sec = (start_frame + max_frames) * self.hop_length / self.sr
                valid_mask = (raw_labels[:, 1] > start_sec) & (raw_labels[:, 0] < end_sec)
                raw_labels = raw_labels[valid_mask].clone()
                if len(raw_labels) > 0:
                    raw_labels[:, 0] = torch.clamp(raw_labels[:, 0] - start_sec, min=0.0)
                    raw_labels[:, 1] = torch.clamp(raw_labels[:, 1] - start_sec, max=end_sec - start_sec)

        return features, (onset_labels, fret_labels), raw_labels, track["audio"]

def get_track_lists():
    print("Mapping audio files to JAMS annotations...")
    ds_map = {"GuitarSet": [], "Synthetic": []}
    
    # 1. GuitarSet
    gs_audio_files = list(glob.iglob(os.path.join(DATASET_BASE_DIR, "GuitarSet", "audio_mono-mic", "*_mic.wav")))
    for audio_path in gs_audio_files:
        basename = os.path.basename(audio_path).replace("_mic.wav", ".jams")
        jams_path = os.path.join(DATASET_BASE_DIR, "GuitarSet", "annotation", basename)
        if os.path.exists(jams_path):
            ds_map["GuitarSet"].append({"audio": audio_path, "jams": jams_path, "type": "GuitarSet"})

    # 2. Synthetic Datasets (Martin, Taylor, Gibson, Luthier)
    try:
        outall_items = set(os.listdir(OUTALL_DIR))
    except Exception:
        outall_items = set()

    for synth_dir in ["martin_finger", "martin_pick", "taylor_finger", "taylor_pick", "gibson_thumb", "luthier_finger", "luthier_pick"]:
        target_dir = os.path.join(DATASET_BASE_DIR, synth_dir)
        if not os.path.exists(target_dir): continue
        
        flac_files = list(glob.iglob(os.path.join(target_dir, "**", "*_nonoise_mono_body.flac"), recursive=True))
        for flac_file in flac_files:
            parent_dir = os.path.basename(os.path.dirname(flac_file))
            if parent_dir in outall_items:
                jams_target_dir = os.path.join(OUTALL_DIR, parent_dir)
                jams_files = [f for f in os.listdir(jams_target_dir) if f.endswith(".jams")]
                if jams_files:
                    ds_map["Synthetic"].append({"audio": flac_file, "jams": os.path.join(jams_target_dir, jams_files[0]), "type": "Synthetic"})
    
    print(f"Mapped {len(ds_map['GuitarSet'])} GuitarSet tracks and {len(ds_map['Synthetic'])} Synthetic tracks.")
    return ds_map

def validate_dataloader_integrity(dataloader, num_batches=10):
    print("\n[Layer 1] Running Dry-Run Validation on DataLoader...")
    for i, batch in enumerate(dataloader):
        if i >= num_batches: break
        features, (onset_labels, fret_labels), lengths, raw_labels, paths = batch
        assert onset_labels.sum().item() > 0, f"🔥 FATAL ERROR: Empty (silent) labels found in {paths}. Parsing might be broken!"
        assert not torch.isnan(features).any(), "🔥 FATAL ERROR: NaN found in CQT features!"
    print("[Layer 1] Dry-Run passed successfully.\n")

def custom_train_one_epoch(model, train_loader, optimizer, criterion, device, specaugment=None):
    model.train()
    total_loss_sum = 0.0
    valid_batches = 0
    
    pbar = tqdm(train_loader, desc="[Training]", unit="batch", leave=False, dynamic_ncols=True)
    for batch in pbar:
        features, (onset_targets, fret_targets), lengths, *_ = batch
        
        # 5. Dynamic Loss Skip: Avoid silently polluting weights on empty batches
        if onset_targets.sum().item() == 0 and fret_targets.sum().item() == 0:
            continue
            
        features = features.to(device)
        
        # Apply GPU SpecAugment if enabled
        if specaugment is not None:
            features = specaugment(features)
            
        onset_targets = onset_targets.to(device)
        fret_targets = fret_targets.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        onset_preds, fret_preds = model(features)
        
        loss, _, _ = criterion(onset_preds, fret_preds, onset_targets, fret_targets, lengths)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss_sum += loss.item()
        valid_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return {"train_total_loss": total_loss_sum / max(1, valid_batches)}

def run_ultimate_finetuning():
    print("=" * 60)
    print(" Starting Ultimate Fine-Tuning ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_map = get_track_lists()
    all_tracks = ds_map["GuitarSet"] + ds_map["Synthetic"]
    dataset = RobustGuitarDataset(all_tracks, augment=True)

    # 3. Weighted Random Sampler (Golden Ratio)
    # Ensure GuitarSet is heavily represented in each batch
    weights = []
    for t in all_tracks:
        weight = 50.0 if t["type"] == "GuitarSet" else 1.0 # 50x multiplier for real data
        weights.append(weight)
        
    # Validation split - using 10% randomly
    val_indices = np.random.choice(len(all_tracks), int(len(all_tracks) * 0.05), replace=False)
    val_subset = Subset(RobustGuitarDataset(all_tracks, augment=False), val_indices)
    
    train_sampler = WeightedRandomSampler(weights, num_samples=3000, replacement=True) # 3000 batches per epoch roughly equivalent to standard
    
    train_loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE_DEFAULT,
        sampler=train_sampler, collate_fn=collate_fn_pad, 
        num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=False, collate_fn=collate_fn_pad, 
        num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True
    )

    # 1. 🚀 Dry-Run Validation
    validate_dataloader_integrity(train_loader, num_batches=3)

    # Building Model and loading run_1 weights
    print(f"Loading Base Model Config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        run_config = json.load(f)
    hyperparams = run_config["hyperparameters_tuned"]
    
    with torch.no_grad():
        out = architecture.TabCNN()(torch.randn(1, 3, config.N_BINS_CQT, 32))
        cnn_out_dim = out.shape[1] * out.shape[2]

    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type="CONFORMER",
        rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams["RNN_LAYERS"],
        rnn_dropout=hyperparams["RNN_DROPOUT"],
        rnn_bidirectional=hyperparams.get("RNN_BIDIRECTIONAL", True),
    )

    model.to(device)

    onset_pos_weight = torch.tensor([hyperparams["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device) if hyperparams.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None
    criterion = loss_functions.CombinedLoss(onset_pos_weight=onset_pos_weight, onset_loss_weight=hyperparams["ONSET_LOSS_WEIGHT"]).to(device)

    # 4. 極小スタート
    optimizer = optim.AdamW(model.parameters(), lr=LR_START, weight_decay=hyperparams["WEIGHT_DECAY"])

    best_f1 = 0.0
    bad_epochs = 0
    
    # 5. Initialize GPU SpecAugment
    gpu_specaugment = torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=40),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=26)
    ).to(device)

    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(f"--- Ultimate Fine-Tuning Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_metrics = custom_train_one_epoch(model, train_loader, optimizer, criterion, device, specaugment=gpu_specaugment)
            
            val_pbar = tqdm(val_loader, desc=f"Evaluating", leave=False, dynamic_ncols=True)
            val_metrics = epoch_processing.evaluate_one_epoch(model, val_pbar, criterion, device, config)
            
            f1_score = val_metrics.get("val_tdr_f1_at_0.5", 0.0)
            log_line = f"Epoch {epoch} | Train Loss: {train_metrics['train_total_loss']:.4f} | Val F1: {f1_score:.4f} | P: {val_metrics.get('val_tdr_precision_at_0.5', 0.0):.4f} | R: {val_metrics.get('val_tdr_recall_at_0.5', 0.0):.4f}"
            print(log_line)
            log_f.write(log_line + "\n")
            log_f.flush()

            # 2. 🛡️ スコア更新・保存判定（自己ベストを更新できない状態が20回続けば終了）
            if f1_score > best_f1:
                # 自己ベスト更新！保存してペナルティ（停滞）をリセット
                best_f1 = f1_score
                bad_epochs = 0
                torch.save(model.state_dict(), NEW_MODEL_PATH)
                print(f"🌟 新しい最高記録 Best F1: {best_f1:.4f} を保存しました！ (停滞カウントリセット)")
            else:
                # 更新できなかった場合は停滞としてカウント
                bad_epochs += 1
                print(f"⚠️ 警告: 最高記録({best_f1:.4f})を更新できませんでした。停滞カウント: {bad_epochs}/{PATIENCE_EARLY_STOP}")
                
                if bad_epochs >= PATIENCE_EARLY_STOP:
                    print(f"🔥 {PATIENCE_EARLY_STOP} Epoch連続で成長・自己記録の更新なし。限界とみなし、アーリーストップを実行します。")
                    break

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_ultimate_finetuning()
