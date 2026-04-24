import os
import argparse
import sys
import io

# Fix CP932 console encoding issues on Windows (with line buffering)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

os.environ["TQDM_ASCII"] = " 123456789#"

import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch import optim
from tqdm import tqdm
import time
import json
import glob
import shutil

# Imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from data_processing.preparation import extract_annotations_from_jams
from data_processing.dataset import create_frame_level_labels
from training import pipeline, loss_functions, epoch_processing
from model import architecture, utils
import config
import torchaudio

parser = argparse.ArgumentParser(description="Train specialized domain models")
parser.add_argument("--dataset", type=str, required=True, help="Name of dataset, e.g., taylor_finger")
args = parser.parse_args()

DATASET_NAME = args.dataset
DATASET_DIR = os.path.join(r"D:\Music\datasets", DATASET_NAME)
OUTALL_DIR = r"D:\Music\all_jams_midi_V2_60000_tracks\outall"
OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")

if DATASET_NAME == "taylor_finger":
    BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "finetuned_martin_finger_model")
elif DATASET_NAME == "luthier_finger":
    BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "finetuned_taylor_finger_model")
elif DATASET_NAME == "martin_pick":
    BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "finetuned_luthier_finger_model")
elif DATASET_NAME == "taylor_pick":
    BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "finetuned_martin_pick_model")
elif DATASET_NAME == "luthier_pick":
    BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "finetuned_taylor_pick_model")
else:
    BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "run_1_baseline_gru_augEnabled")

RUN_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", f"finetuned_{DATASET_NAME}_model")

os.makedirs(RUN_DIR, exist_ok=True)
if not os.path.exists(os.path.join(RUN_DIR, "run_configuration.json")):
    shutil.copy(os.path.join(BASELINE_DIR, "run_configuration.json"), os.path.join(RUN_DIR, "run_configuration.json"))

# Determine Resume or Start new
EXISTING_MODEL_PATH = os.path.join(RUN_DIR, "best_model.pth")
LOG_PATH = os.path.join(RUN_DIR, "training_log.txt")

if os.path.exists(EXISTING_MODEL_PATH):
    print(f"[*] Found existing model for {DATASET_NAME}. Resuming...")
    MODEL_LOAD_PATH = EXISTING_MODEL_PATH
else:
    print(f"[*] Starting fresh training for {DATASET_NAME} from Baseline.")
    MODEL_LOAD_PATH = os.path.join(BASELINE_DIR, "best_model.pth")

MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
CONFIG_PATH = os.path.join(RUN_DIR, "run_configuration.json")

TOTAL_EPOCHS = 300
RESUME_LR = 6e-5
START_EPOCH = 1
BEST_METRIC_VAL = 0.0

# Try to parse log to find the last epoch and best F1
if os.path.exists(LOG_PATH):
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        found_start_epoch = False
        for line in reversed(lines):
            line = line.strip()
            if "Best F1 (val_tdr_f1_at_0.5):" in line:
                # e.g., 🌟 新しい最高記録 Best F1 (val_tdr_f1_at_0.5): 0.7758 を保存しました！
                parts = line.split("):")
                if len(parts) > 1:
                    score_str = parts[1].split()[0]
                    BEST_METRIC_VAL = max(BEST_METRIC_VAL, float(score_str))
            if not found_start_epoch and line.startswith("Epoch ") and "|" in line:
                # e.g., Epoch 20 | Train Loss: 0.5876...
                ep_part = line.split("|")[0].strip().split()
                if len(ep_part) == 2:
                    last_ep = int(ep_part[1])
                    START_EPOCH = last_ep + 1
                    found_start_epoch = True
    except Exception as e:
        print(f"Warning: Could not parse log file {e}")

REMAINING_EPOCHS = TOTAL_EPOCHS - START_EPOCH + 1

if REMAINING_EPOCHS <= 0:
    print(f"[*] Training for {DATASET_NAME} already finished.")
    sys.exit(0)

class DomainSynthDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.root_dir = root_dir
        self.augment = augment
        self.sr = config.SAMPLE_RATE
        self.hop_length = config.HOP_LENGTH
        
        if self.augment:
            self.specaugment = torch.nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=40),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=26)
            )
            
        print(f"Skanowanie katalogu: {root_dir}")
        self.tracks = []
        try:
            outall_items = set(os.listdir(OUTALL_DIR))
        except Exception:
            outall_items = set()
            
        for flac_file in glob.iglob(os.path.join(root_dir, "**", "*_nonoise_mono_body.flac"), recursive=True):
            parent_dir_name = os.path.basename(os.path.dirname(flac_file))
            if parent_dir_name in outall_items:
                jams_target_dir = os.path.join(OUTALL_DIR, parent_dir_name)
                try:
                    expected_jams = [f for f in os.listdir(jams_target_dir) if f.endswith(".jams")]
                    if expected_jams:
                        self.tracks.append({
                            "flac": flac_file,
                            "jams": os.path.join(jams_target_dir, expected_jams[0])
                        })
                except FileNotFoundError:
                    pass
        print(f"Found {len(self.tracks)} valid tracks for {DATASET_NAME}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        try:
            track = self.tracks[idx]
            cache_path = track["flac"] + ".cqt.pt"
            if os.path.exists(cache_path):
                try:
                    features = torch.load(cache_path, weights_only=False)
                except Exception:
                    features = self._compute_and_cache(track, cache_path)
            else:
                features = self._compute_and_cache(track, cache_path)
                
            if self.augment:
                features = features.unsqueeze(0)
                features = self.specaugment(features)
                features = features.squeeze(0)
                
            raw_notes = extract_annotations_from_jams(track["jams"])
            if len(raw_notes) > 0:
                raw_labels = torch.tensor(raw_notes, dtype=torch.float32)
            else:
                raw_labels = torch.empty((0, 5)) 

            onset_labels, fret_labels = create_frame_level_labels(raw_labels, features, self.hop_length, self.sr, config.MAX_FRETS)
            
            max_frames = 430
            total_frames = features.shape[1]
            if total_frames > max_frames:
                start_frame = np.random.randint(0, total_frames - max_frames)
                features = features[:, start_frame : start_frame + max_frames]
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
            
            return features, (onset_labels, fret_labels), raw_labels, track["flac"]
        except Exception as e:
            print(f"Skipping corrupted track {idx}: {e}")
            new_idx = np.random.randint(len(self.tracks))
            return self.__getitem__(new_idx)
        
    def _compute_and_cache(self, track, cache_path):
        y_tensor, sr_load = torchaudio.load(track["flac"])
        if sr_load != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_load, new_freq=self.sr)
            y_tensor = resampler(y_tensor)
        y = y_tensor.squeeze(0).numpy()
        num_frames = int(np.ceil(len(y) / self.hop_length))
        cqt_spec = librosa.cqt(
            y=y, sr=self.sr, hop_length=self.hop_length,
            fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
        )
        log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        if log_cqt_spec.shape[1] > num_frames:
            log_cqt_spec = log_cqt_spec[:, :num_frames]
        elif log_cqt_spec.shape[1] < num_frames:
            pad_width = num_frames - log_cqt_spec.shape[1]
            log_cqt_spec = np.pad(log_cqt_spec, ((0, 0), (0, pad_width)), mode='constant')
        features = torch.tensor(log_cqt_spec, dtype=torch.float32)
        torch.save(features, cache_path)
        return features


def resume_training():
    print("=" * 60)
    print(f"Starting {DATASET_NAME} Fine-Tuning")
    print(f"Resuming at Epoch: {START_EPOCH}, Best F1 so far: {BEST_METRIC_VAL:.4f}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        run_config = json.load(f)
    hyperparams = run_config["hyperparameters_tuned"]
    
    print("\n[1] Preparing Data Loader...")
    ds = DomainSynthDataset(DATASET_DIR, augment=True)
    if len(ds) == 0:
        print("No valid tracks found. Exiting.")
        sys.exit(1)
        
    from torch.utils.data import Subset
    EPOCH_SIZE = min(4000, len(ds))
    indices = np.random.choice(len(ds), EPOCH_SIZE, replace=False)
    epoch_subset_train = Subset(ds, indices)
    val_indices = np.random.choice(len(ds), int(EPOCH_SIZE * 0.1), replace=False)
    epoch_subset_val = Subset(ds, val_indices)
        
    train_loader = DataLoader(epoch_subset_train, batch_size=8, shuffle=True, collate_fn=collate_fn_pad, num_workers=6, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(epoch_subset_val, batch_size=8, shuffle=False, collate_fn=collate_fn_pad, num_workers=6, pin_memory=True, prefetch_factor=2)

    # 2. Build model and load weights
    print(f"\n[2] Loading model from checkpoint... {MODEL_LOAD_PATH}")
    temp_cnn = architecture.TabCNN()
    with torch.no_grad():
        out = temp_cnn(torch.randn(1, 1, config.N_BINS_CQT, 32))
        cnn_out_dim = out.shape[1] * out.shape[2]
    del temp_cnn

    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hyperparams.get("RNN_TYPE", "GRU"),
        rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams["RNN_LAYERS"],
        rnn_dropout=hyperparams["RNN_DROPOUT"],
        rnn_bidirectional=hyperparams.get("RNN_BIDIRECTIONAL", True),
    )

    state_dict = torch.load(MODEL_LOAD_PATH, map_location=device, weights_only=False)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)

    # 3. Setup training components
    onset_pos_weight = torch.tensor([hyperparams["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device) if hyperparams.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None
    criterion = loss_functions.CombinedLoss(onset_pos_weight=onset_pos_weight, onset_loss_weight=hyperparams["ONSET_LOSS_WEIGHT"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=RESUME_LR, weight_decay=hyperparams["WEIGHT_DECAY"])

    scheduler_mode = "max" if "loss" not in config.CHECKPOINT_METRIC_DEFAULT.lower() else "min"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=hyperparams["SCHEDULER_FACTOR"], patience=hyperparams["SCHEDULER_PATIENCE"])

    # Resume optimizer state if we are recovering from a crash
    opt_path = os.path.join(RUN_DIR, "latest_optimizer.pth")
    if START_EPOCH > 1 and os.path.exists(opt_path):
        try:
            print(f"[*] Loading optimizer state from {opt_path}")
            opt_state_dict = torch.load(opt_path, map_location=device, weights_only=False)
            if "optimizer" in opt_state_dict:
                optimizer.load_state_dict(opt_state_dict["optimizer"])
            if scheduler and opt_state_dict.get("scheduler"):
                scheduler.load_state_dict(opt_state_dict["scheduler"])
            print("[*] Optimizer and Scheduler states loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}")

    # 4. Training loop
    print(f"\n[3] Starting Fine-tuning for {DATASET_NAME}...")
    best_metric_val = BEST_METRIC_VAL
    epochs_without_improvement = 0
    early_stop_patience = config.EARLY_STOPPING_PATIENCE_DEFAULT

    training_history = {
        "train_total_loss": [], "val_total_loss": [], "lr": [], "val_tdr_f1_at_0.5": [], "val_tdr_precision_at_0.5": [], "val_tdr_recall_at_0.5": [], "val_onset_f1_event_at_0.5": [], "val_mpe_f1": []
    }

    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n\n{'='*50}\n")
        log_f.write(f"--- STARTED at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_f.write(f"{'='*50}\n")

        for epoch_offset in range(REMAINING_EPOCHS):
            current_epoch = START_EPOCH + epoch_offset
            epoch_desc = f"Epoka {current_epoch}/{TOTAL_EPOCHS}"

            train_pbar = tqdm(train_loader, desc=f"{epoch_desc} [Trening]", unit="batch", leave=False, dynamic_ncols=True)
            train_metrics = epoch_processing.train_one_epoch(model, train_pbar, optimizer, criterion, device)
            
            if "train_total_loss" in training_history:
                training_history["train_total_loss"].append(train_metrics["train_total_loss"])

            val_pbar = tqdm(val_loader, desc=f"{epoch_desc} [Walidacja]", unit="batch", leave=False, dynamic_ncols=True)
            val_metrics = epoch_processing.evaluate_one_epoch(model, val_pbar, criterion, device, config)

            for key, value in val_metrics.items():
                if key in training_history:
                    training_history[key].append(value)

            f1_score = val_metrics.get('val_tdr_f1_at_0.5', 0.0)
            p_val = val_metrics.get('val_tdr_precision_at_0.5', 0.0)
            r_val = val_metrics.get('val_tdr_recall_at_0.5', 0.0)
            log_line = f"Epoch {current_epoch} | Train Loss: {train_metrics['train_total_loss']:.4f} | Val F1: {f1_score:.4f} | P: {p_val:.4f} | R: {r_val:.4f}"
            print(log_line)
            log_f.write(log_line + "\n")

            effective_metric = f"{config.CHECKPOINT_METRIC_DEFAULT}_at_0.5" if config.CHECKPOINT_METRIC_DEFAULT not in val_metrics else config.CHECKPOINT_METRIC_DEFAULT
            metric_val = val_metrics.get(effective_metric, float("-inf"))
            if scheduler: scheduler.step(metric_val)

            is_improved = metric_val >= best_metric_val
            if is_improved and metric_val != float("-inf"):
                epochs_without_improvement = 0
                best_metric_val = metric_val
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                
                # Optimizer Backup
                opt_state = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                }
                torch.save(opt_state, os.path.join(RUN_DIR, "latest_optimizer.pth"))
                
                msg = f"新しい最高記録 Best F1 ({effective_metric}): {best_metric_val:.4f} を保存しました！ (停滞カウントリセット)"
                print(msg)
                log_f.write(msg + "\n")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    msg = f"\n    Early stopping after {epochs_without_improvement} epochs without improvement."
                    print(msg)
                    log_f.write(msg + "\n")
                    break

            if current_epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(RUN_DIR, f"model_epoch_{current_epoch}.pth"))
                # Save optimizer state too for checkpoints
                opt_backup = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                }
                torch.save(opt_backup, os.path.join(RUN_DIR, f"opt_epoch_{current_epoch}.pth"))
                msg_backup = f"\n[定期防衛バックアップ] Epoch {current_epoch}の単独ファイルを保存しました: model_epoch_{current_epoch}.pth\n"
                print(msg_backup)
                log_f.write(msg_backup + "\n")
            log_f.flush()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    resume_training()
