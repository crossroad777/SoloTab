import os
os.environ["TQDM_ASCII"] = " 123456789#"

import sys
import io
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch import optim
from tqdm import tqdm
import time
import json
import glob

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
from evaluation import performance_metrics
from vizualization import plotting
import config
import torchaudio

# Configure paths
MARTIN_FINGER_DIR = r"D:\Music\datasets\martin_finger"
MARTIN_PICK_DIR = r"D:\Music\datasets\martin_pick"
OUTALL_DIR = r"D:\Music\all_jams_midi_V2_60000_tracks\outall"
OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")

BASELINE_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "baseline_model")
RUN_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output", "finetuned_martin_model")

os.makedirs(RUN_DIR, exist_ok=True)
import shutil
if not os.path.exists(os.path.join(RUN_DIR, "run_configuration.json")):
    shutil.copy(os.path.join(BASELINE_DIR, "run_configuration.json"), os.path.join(RUN_DIR, "run_configuration.json"))

MODEL_LOAD_PATH = os.path.join(BASELINE_DIR, "best_model.pth")
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
CONFIG_PATH = os.path.join(RUN_DIR, "run_configuration.json")
LOG_PATH = os.path.join(RUN_DIR, "training_log.txt")

START_EPOCH = 1
TOTAL_EPOCHS = 300
RESUME_LR = 0.0001
REMAINING_EPOCHS = TOTAL_EPOCHS - START_EPOCH + 1

# [Exact Copy of MartinSynthDataset from train_martin_only.py]
class MartinSynthDataset(Dataset):
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
            
        print(f"Skanowanie katalogu i mapowanie JAMS: {root_dir}")
        self.tracks = []
        try:
            outall_items = set(os.listdir(OUTALL_DIR))
        except Exception as e:
            outall_items = set()
            
        count_matched = 0
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
                        count_matched += 1
                except FileNotFoundError:
                    pass
        print(f"Found {len(self.tracks)} valid tracks linked to JAMS for {root_dir}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
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
        print(f"DEBUG GETITEM: items in '{os.path.basename(track['jams'])}': {len(raw_notes)}")
        if len(raw_notes) > 0:
            raw_labels = torch.tensor(raw_notes, dtype=torch.float32)
        else:
            raw_labels = torch.empty((0, 5)) 

        onset_labels, fret_labels = create_frame_level_labels(
            raw_labels, features, self.hop_length, self.sr, config.MAX_FRETS
        )
        
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
        
        output_tuple = (onset_labels, fret_labels)
        return features, output_tuple, raw_labels, track["flac"]
        
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
    print("Resuming Martin Fine-Tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        run_config = json.load(f)
    hyperparams = run_config["hyperparameters_tuned"]
    
    print("\n[1] Preparing Martin Data Loader (Step 1: martin_finger ONLY)...")
    ds_finger = MartinSynthDataset(MARTIN_FINGER_DIR, augment=True)
    
    combined_train = ds_finger
    
    from torch.utils.data import Subset
    EPOCH_SIZE = 4000
    if len(combined_train) > EPOCH_SIZE:
        indices = np.random.choice(len(combined_train), EPOCH_SIZE, replace=False)
        epoch_subset_train = Subset(combined_train, indices)
        val_indices = np.random.choice(len(combined_train), int(EPOCH_SIZE * 0.1), replace=False)
        epoch_subset_val = Subset(combined_train, val_indices)
    else:
        epoch_subset_train = combined_train
        epoch_subset_val = combined_train
        
    train_loader = DataLoader(
        epoch_subset_train, batch_size=8,
        shuffle=True, collate_fn=collate_fn_pad, num_workers=6, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        epoch_subset_val, batch_size=8,
        shuffle=False, collate_fn=collate_fn_pad, num_workers=6, pin_memory=True, prefetch_factor=2
    )

    # 2. Build model and load weights
    print(f"\n[2] Loading model from checkpoint... {MODEL_LOAD_PATH}")
    temp_cnn = architecture.TabCNN()
    with torch.no_grad():
        dummy = torch.randn(1, 1, config.N_BINS_CQT, 32)
        out = temp_cnn(dummy)
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
    print(f"  Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Setup training components
    onset_pos_weight = (
        torch.tensor([hyperparams["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device)
        if hyperparams.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None
    )
    criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pos_weight,
        onset_loss_weight=hyperparams["ONSET_LOSS_WEIGHT"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=RESUME_LR,
        weight_decay=hyperparams["WEIGHT_DECAY"],
    )

    scheduler_mode = "max" if "loss" not in config.CHECKPOINT_METRIC_DEFAULT.lower() else "min"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=hyperparams["SCHEDULER_FACTOR"],
        patience=hyperparams["SCHEDULER_PATIENCE"],
    )

    # 4. Training loop with resume
    print(f"\n[3] Resuming training from epoch {START_EPOCH}...")
    checkpoint_metric = config.CHECKPOINT_METRIC_DEFAULT
    # Note: Resetting best_metric_val to 0.0 so the model can actually save improvements on the Martin dataset.
    best_metric_val = 0.0
    epochs_without_improvement = 0
    early_stop_patience = config.EARLY_STOPPING_PATIENCE_DEFAULT

    training_history = {
        "train_total_loss": [], "val_total_loss": [], "lr": [],
        "val_tdr_f1_at_0.5": [], "val_tdr_precision_at_0.5": [], "val_tdr_recall_at_0.5": [],
        "val_onset_f1_event_at_0.5": [], "val_mpe_f1": [],
    }

    start_time = time.time()
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n\n{'='*50}\n")
        log_f.write(f"--- RESUMED at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
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

            current_lr = optimizer.param_groups[0]["lr"]
            f1_score = val_metrics.get('val_tdr_f1_at_0.5', 0.0)
            p_val = val_metrics.get('val_tdr_precision_at_0.5', 0.0)
            r_val = val_metrics.get('val_tdr_recall_at_0.5', 0.0)
            log_line = f"Epoch {current_epoch} | Train Loss: {train_metrics['train_total_loss']:.4f} | Val F1: {f1_score:.4f} | P: {p_val:.4f} | R: {r_val:.4f}"
            print(log_line)
            log_f.write(log_line + "\n")
            log_f.flush()

            effective_metric = checkpoint_metric
            if effective_metric not in val_metrics:
                effective_metric = f"{effective_metric}_at_0.5"

            metric_val = val_metrics.get(effective_metric, float("-inf"))
            if scheduler:
                scheduler.step(metric_val)

            is_improved = metric_val >= best_metric_val
            if is_improved and metric_val != float("-inf"):
                epochs_without_improvement = 0
                best_metric_val = metric_val
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                msg = f"🌟 新しい最高記録 Best F1 ({effective_metric}): {best_metric_val:.4f} を保存しました！ (停滞カウントリセット)"
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    msg = f"\n    Early stopping after {epochs_without_improvement} epochs without improvement."
                    print(msg)
                    log_f.write(msg + "\n")
                    break

            # --- ユーザー様の資産（時間とコスト）を守るための絶対防衛：20エポック定期セーブ ---
            if current_epoch % 20 == 0:
                epoch_save_path = os.path.join(RUN_DIR, f"model_epoch_{current_epoch}.pth")
                torch.save(model.state_dict(), epoch_save_path)
                msg_backup = f"\n💿 [定期防衛バックアップ] ユーザー様の時間を保護するため、決して上書きされないEpoch {current_epoch}の単独ファイルを保存しました: model_epoch_{current_epoch}.pth\n"
                print(msg_backup)
                log_f.write(msg_backup + "\n")
                log_f.flush()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    resume_training()
