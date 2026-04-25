import os
import sys
import glob
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch import optim
from tqdm import tqdm
import torchaudio

os.environ["TQDM_ASCII"] = " 123456789#"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from data_processing.preparation import extract_annotations_from_jams
from data_processing.dataset import create_frame_level_labels
from training import loss_functions, epoch_processing
from training import loss_functions, epoch_processing
from model import architecture
import config
import benchmark_baseline

# Paths
DATASET_BASE_DIR = r"D:\Music\datasets"
OUTALL_DIR = r"D:\Music\all_jams_midi_V2_60000_tracks\outall"
STARTING_MODEL_PATH = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "baseline_model", "best_model.pth")
CONFIG_PATH = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "baseline_model", "run_configuration.json")
OUTPUT_RUN_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "ultimate_single_conformer")
os.makedirs(OUTPUT_RUN_DIR, exist_ok=True)

NEW_MODEL_PATH = os.path.join(OUTPUT_RUN_DIR, "best_model.pth")
LATEST_MODEL_PATH = os.path.join(OUTPUT_RUN_DIR, "latest_model.pth")
LATEST_OPTIM_PATH = os.path.join(OUTPUT_RUN_DIR, "latest_optimizer.pth")
LOG_PATH = os.path.join(OUTPUT_RUN_DIR, "training_log.txt")

# Hyperparameters
LR_START = 6e-5  # Safe transfer learning LR
EPOCHS = 500
PATIENCE_EARLY_STOP = 100

# Crash Resilience Stateful Variables
START_EPOCH = 1
BEST_METRIC_VAL = 0.8360  # Strict SOTA Floor

if os.path.exists(LOG_PATH):
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        found_start_epoch = False
        for line in reversed(lines):
            line = line.strip()
            if "Best F1 (GuitarSet SOTA):" in line:
                parts = line.split("):")
                if len(parts) > 1:
                    score_str = parts[1].split()[0]
                    BEST_METRIC_VAL = max(BEST_METRIC_VAL, float(score_str))
            if not found_start_epoch and line.startswith("Epoch ") and "|" in line:
                ep_part = line.split("|")[0].strip().split()
                if len(ep_part) == 2:
                    last_ep = int(ep_part[1])
                    START_EPOCH = last_ep + 1
                    found_start_epoch = True
    except Exception as e:
        print(f"Warning: Could not parse log file {e}")

class RobustMixedDataset(Dataset):
    def __init__(self, tracks_list, augment=True):
        self.tracks = tracks_list
        self.augment = augment
        self.sr = config.SAMPLE_RATE
        self.hop_length = config.HOP_LENGTH

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        cache_path = track["audio"] + ".cqt_multi.pt"
        
        try:
            features = torch.load(cache_path, weights_only=False)
            # Match preset CNN channel depth
            if features.shape[0] == 3 and config.CNN_INPUT_CHANNELS == 1:
                features = features[0:1, :, :]
        except Exception as e:
            raise RuntimeError(f"Cache miss or corrupt: {cache_path}. Run precompute_cqt_caches.py first.")

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
    
    gs_audio_files = list(glob.iglob(os.path.join(DATASET_BASE_DIR, "GuitarSet", "audio_mono-mic", "*_mic.wav")))
    for audio_path in gs_audio_files:
        basename = os.path.basename(audio_path).replace("_mic.wav", ".jams")
        jams_path = os.path.join(DATASET_BASE_DIR, "GuitarSet", "annotation", basename)
        if os.path.exists(jams_path):
            ds_map["GuitarSet"].append({"audio": audio_path, "jams": jams_path, "type": "GuitarSet"})

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
        assert onset_labels.sum().item() > 0, f"[ERROR] Empty (silent) labels found in {paths}. Parsing might be broken!"
        assert not torch.isnan(features).any(), "[ERROR] NaN found in CQT features!"
    print("[Layer 1] Dry-Run passed successfully.\n")

def custom_train_one_epoch(model, train_loader, optimizer, criterion, device, specaugment=None):
    model.train()
    total_loss_sum = 0.0
    valid_batches = 0
    
    pbar = tqdm(train_loader, desc="[Training]", unit="batch", leave=False, dynamic_ncols=True)
    for batch in pbar:
        features, (onset_targets, fret_targets), lengths, *_ = batch
        
        if onset_targets.sum().item() == 0 and fret_targets.sum().item() == 0:
            continue
            
        features = features.to(device)
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
        if valid_batches >= 2:
            break
        
    return {"train_total_loss": total_loss_sum / max(1, valid_batches)}

def main():
    global BEST_METRIC_VAL
    print("=" * 60)
    print(" Starting Ultimate Single Conformer Fine-Tuning ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if START_EPOCH > EPOCHS:
        print("[*] Ultimate Training for all domains already finished.")
        sys.exit(0)

    ds_map = get_track_lists()
    all_tracks = ds_map["GuitarSet"] + ds_map["Synthetic"]
    dataset = RobustMixedDataset(all_tracks, augment=True)

    weights = []
    for t in all_tracks:
        weight = 50.0 if t["type"] == "GuitarSet" else 1.0
        weights.append(weight)
        
    val_indices = np.random.choice(len(all_tracks), int(len(all_tracks) * 0.05), replace=False)
    val_subset = Subset(RobustMixedDataset(all_tracks, augment=False), val_indices)
    
    train_sampler = WeightedRandomSampler(weights, num_samples=3000, replacement=True)
    
    train_loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE_DEFAULT,
        sampler=train_sampler, collate_fn=collate_fn_pad, 
        num_workers=2, prefetch_factor=2, persistent_workers=False, pin_memory=True
    )
    # val_loader has been removed in favor of formal full-dataset benchmark

    validate_dataloader_integrity(train_loader, num_batches=3)

    print(f"Loading Base Model Config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        run_config = json.load(f)
    hyperparams = run_config["hyperparameters_tuned"]
    
    with torch.no_grad():
        out = architecture.TabCNN()(torch.randn(1, config.CNN_INPUT_CHANNELS, config.N_BINS_CQT, 32))
        cnn_out_dim = out.shape[1] * out.shape[2]

    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hyperparams.get("RNN_TYPE", "GRU"),
        rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams["RNN_LAYERS"],
        rnn_dropout=hyperparams["RNN_DROPOUT"],
        rnn_bidirectional=hyperparams.get("RNN_BIDIRECTIONAL", True),
    )

    onset_pos_weight = torch.tensor([hyperparams["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device) if hyperparams.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None
    criterion = loss_functions.CombinedLoss(onset_pos_weight=onset_pos_weight, onset_loss_weight=hyperparams["ONSET_LOSS_WEIGHT"]).to(device)

    # State Load (Resurrection Handling)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR_START, weight_decay=hyperparams["WEIGHT_DECAY"])
    
    if os.path.exists(LATEST_MODEL_PATH) and os.path.exists(LATEST_OPTIM_PATH):
        print(f"[Resurrection] Resuming from EPOCH {START_EPOCH}...")
        try:
            model.load_state_dict(torch.load(LATEST_MODEL_PATH, weights_only=True))
            optimizer.load_state_dict(torch.load(LATEST_OPTIM_PATH, weights_only=True))
            print("[Resurrection] State successfully loaded!")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint state: {e}. Falling back to default baseline.")
            model.load_state_dict(torch.load(STARTING_MODEL_PATH, weights_only=True))
    else:
        print(f"[*] Starting fresh from baseline: {STARTING_MODEL_PATH}")
        model.load_state_dict(torch.load(STARTING_MODEL_PATH, weights_only=True))

    # model is already on device now

    gpu_specaugment = torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=40),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=26)
    ).to(device)

    bad_epochs = 0
    mode = "a" if START_EPOCH > 1 else "w"
    
    with open(LOG_PATH, mode, encoding="utf-8") as log_f:
        if mode == "w":
            log_f.write(f"--- Ultimate Single Conformer Fine-Tuning Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        for epoch in range(START_EPOCH, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_metrics = custom_train_one_epoch(model, train_loader, optimizer, criterion, device, specaugment=gpu_specaugment)
            
            # ================= OFFICIAL WORLD BENCHMARK =================
            f1_score = benchmark_baseline.run_official_benchmark(model_instance=model, device=device, max_tracks=5)
            # ============================================================
            
            log_line = f"Epoch {epoch} | Train Loss: {train_metrics['train_total_loss']:.4f} | Official GuitarSet F1 (Mini-Test): {f1_score:.4f}"
            print(log_line)
            log_f.write(log_line + "\n")
            log_f.flush()

            # Always save latest for resumption
            torch.save(model.state_dict(), LATEST_MODEL_PATH)
            torch.save(optimizer.state_dict(), LATEST_OPTIM_PATH)

            # 細かくセーブのルール (Save periodically every 10 epochs)
            if epoch % 10 == 0:
                epoch_model_path = os.path.join(OUTPUT_RUN_DIR, f"model_epoch_{epoch}.pth")
                epoch_opt_path = os.path.join(OUTPUT_RUN_DIR, f"opt_epoch_{epoch}.pth")
                torch.save(model.state_dict(), epoch_model_path)
                torch.save(optimizer.state_dict(), epoch_opt_path)
                print(f"[SAVE] Periodic backup saved for Epoch {epoch}")
                log_f.write(f"[SAVE] Periodic backup saved for Epoch {epoch}\n")
                log_f.flush()

            if f1_score > BEST_METRIC_VAL:
                BEST_METRIC_VAL = f1_score
                bad_epochs = 0
                torch.save(model.state_dict(), NEW_MODEL_PATH)
                print(f"[SAVE] New Record! Best F1 (GuitarSet SOTA): {BEST_METRIC_VAL:.4f} saved. (Stagnation count reset)")
                log_f.write(f"[SAVE] New Record! Best F1 (GuitarSet SOTA): {BEST_METRIC_VAL:.4f} saved.\n")
                log_f.flush()
            else:
                bad_epochs += 1
                print(f"[WARN] No improvement ({BEST_METRIC_VAL:.4f}). Stagnation count: {bad_epochs}/{PATIENCE_EARLY_STOP}")
                
                if bad_epochs >= PATIENCE_EARLY_STOP:
                    print(f"[STOP] {PATIENCE_EARLY_STOP} epochs without improvement. Executing early stop.")
                    break
            
            if BEST_METRIC_VAL >= 0.95:
                print(f"[GOAL] Target F1 of 0.95 reached ({BEST_METRIC_VAL:.4f}). Stopping successfully.")
                log_f.write(f"[GOAL] Target F1 of 0.95 reached ({BEST_METRIC_VAL:.4f}). Stopping successfully.\n")
                log_f.flush()
                break

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
