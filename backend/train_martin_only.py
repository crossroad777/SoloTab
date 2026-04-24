import os
os.environ["TQDM_ASCII"] = " 123456789#"

import sys
import io
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pickle
import glob

# Fix CP932 console encoding issues on Windows (moved to __main__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from training import pipeline
import config
from data_processing.dataset import _augment_reverb, _augment_eq, _augment_clipping, _augment_add_noise

# ---------------------------------------------------------
# CONSTANTS & CONFIG FOR MARTIN ONLY
# ---------------------------------------------------------

MARTIN_FINGER_DIR = r"D:\Music\datasets\martin_finger"
MARTIN_PICK_DIR = r"D:\Music\datasets\martin_pick"
OUTALL_DIR = r"D:\Music\all_jams_midi_V2_60000_tracks\outall"
OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")

# 選択された [B] 案の実装: Run 3 のベストモデルを初期状態として使用する
RESUME_CHECKPOINT_PATH = os.path.join(OUTPUT_BASE_DIR, "training_output", "run_3_run_3_augDisabled", "best_model.pth")

from data_processing.preparation import extract_annotations_from_jams
from data_processing.dataset import create_frame_level_labels

import torchaudio

class MartinSynthDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.root_dir = root_dir
        self.augment = augment
        self.sr = config.SAMPLE_RATE # 22050
        self.hop_length = config.HOP_LENGTH # 512
        
        # SpecAugment parameters
        if self.augment:
            self.specaugment = torch.nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=40),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=26)
            )
            
        print(f"Skanowanie katalogu i mapowanie JAMS: {root_dir}")
        self.tracks = []
        
        print(f"  > Buforowanie struktury katalogu {OUTALL_DIR}...")
        try:
            outall_items = set(os.listdir(OUTALL_DIR))
        except Exception as e:
            print(f"Błąd czytania {OUTALL_DIR}: {e}")
            outall_items = set()
            
        count_matched = 0
        for flac_file in glob.iglob(os.path.join(root_dir, "**", "*_nonoise_mono_body.flac"), recursive=True):
            parent_dir_name = os.path.basename(os.path.dirname(flac_file))
            
            if parent_dir_name in outall_items:
                jams_target_dir = os.path.join(OUTALL_DIR, parent_dir_name)
                # Zamiast glob, szukaj wprost plików
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
                    
        print(f"Znaleziono {count_matched} poprawnych ścieżek powiązanych z JAMS dla {root_dir}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        cache_path = track["flac"] + ".cqt.pt"
        if os.path.exists(cache_path):
            # Fast path: Load strictly from disk
            try:
                features = torch.load(cache_path, weights_only=True)
            except Exception:
                # Fallback if cache is corrupted
                features = self._compute_and_cache(track, cache_path)
        else:
            # Slow path: Compute CQT and save it
            features = self._compute_and_cache(track, cache_path)
            
        if self.augment:
            # SpecAugment expects >= 3D tensor (..., freq, time) or (channels, freq, time)
            # features is currently (168, time). Add fake channel dim.
            features = features.unsqueeze(0)
            features = self.specaugment(features)
            features = features.squeeze(0)
            
        raw_notes = extract_annotations_from_jams(track["jams"])
        if len(raw_notes) > 0:
            raw_labels = torch.tensor(raw_notes, dtype=torch.float32)
        else:
            raw_labels = torch.empty((0, 5)) 

        onset_labels, fret_labels = create_frame_level_labels(
            raw_labels,
            features,
            self.hop_length,
            self.sr,
            config.MAX_FRETS
        )
        
        # OOM Prevention: Chunk into ~10 second sequences (430 frames)
        max_frames = 430
        total_frames = features.shape[1]
        if total_frames > max_frames:
            start_frame = np.random.randint(0, total_frames - max_frames)
            features = features[:, start_frame : start_frame + max_frames]
            onset_labels = onset_labels[start_frame : start_frame + max_frames, :]
            fret_labels = fret_labels[start_frame : start_frame + max_frames, :]
            
            # Slice and shift raw_labels to match the chosen window
            if len(raw_labels) > 0:
                start_sec = start_frame * self.hop_length / self.sr
                end_sec = (start_frame + max_frames) * self.hop_length / self.sr
                
                valid_mask = (raw_labels[:, 1] > start_sec) & (raw_labels[:, 0] < end_sec)
                raw_labels = raw_labels[valid_mask].clone()
                if len(raw_labels) > 0:
                    raw_labels[:, 0] = torch.clamp(raw_labels[:, 0] - start_sec, min=0.0)
                    raw_labels[:, 1] = torch.clamp(raw_labels[:, 1] - start_sec, max=end_sec - start_sec)
        
        output_tuple = (onset_labels, fret_labels)
        track_id = track["flac"]
        
        return features, output_tuple, raw_labels, track_id
        
    def _compute_and_cache(self, track, cache_path):
        y_tensor, sr_load = torchaudio.load(track["flac"])
        if sr_load != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_load, new_freq=self.sr)
            y_tensor = resampler(y_tensor)
        y = y_tensor.squeeze(0).numpy() # Shape (samples,)
            
        audio_dur = len(y) / self.sr
        num_frames = int(np.ceil(len(y) / self.hop_length))
        
        cqt_spec = librosa.cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=config.FMIN_CQT,
            n_bins=config.N_BINS_CQT,
            bins_per_octave=config.BINS_PER_OCTAVE_CQT
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


def run_experiment():
    print("\n[1] Preparing Martin ONLY DataLoader (with JAMS)...")
    
    ds_finger = MartinSynthDataset(MARTIN_FINGER_DIR, augment=True)
    ds_pick = MartinSynthDataset(MARTIN_PICK_DIR, augment=True)
    
    combined_train = ConcatDataset([ds_finger, ds_pick])
    
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
        epoch_subset_train,
        batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=True,
        collate_fn=collate_fn_pad,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        epoch_subset_val,
        batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=False,
        collate_fn=collate_fn_pad,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2
    )

    print("\n[2] Starting Fine-Tuning Execution...")
    os.environ["RESUME_CHECKPOINT"] = RESUME_CHECKPOINT_PATH
    print(f"Loaded pretrained weights from {RESUME_CHECKPOINT_PATH} (strict=False)")
    
    result = pipeline.process_single_hyperparameter_run(
        run_id=4,
        hyperparams_combo={
            "RUN_DESCRIPTION": "Martin_Only_FineTuning",
            "RNN_TYPE": config.RNN_TYPE_DEFAULT,
            "RNN_HIDDEN_SIZE": config.RNN_HIDDEN_SIZE_DEFAULT,
            "RNN_LAYERS": config.RNN_LAYERS_DEFAULT,
            "RNN_DROPOUT": config.RNN_DROPOUT_DEFAULT,
            "LEARNING_RATE_INIT": 1e-4,  # Standard learning rate
            "WEIGHT_DECAY": config.WEIGHT_DECAY_DEFAULT,
            "SCHEDULER_FACTOR": config.SCHEDULER_FACTOR_DEFAULT,
            "SCHEDULER_PATIENCE": config.SCHEDULER_PATIENCE_DEFAULT,
            "ONSET_POS_WEIGHT_MANUAL_VALUE": config.ONSET_POS_WEIGHT_MANUAL_VALUE_DEFAULT,
            "ONSET_LOSS_WEIGHT": config.ONSET_LOSS_WEIGHT_DEFAULT,
            "RESUME_CHECKPOINT_PATH": RESUME_CHECKPOINT_PATH
        },
        current_augmentation_params=config.DATASET_TRAIN_AUGMENTATION_PARAMS,
        config_obj=config,
        main_artifacts_dir=os.path.join(OUTPUT_BASE_DIR, "training_output"),
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader
    )
    print(f"\nPipeline Return Status: {result}")

if __name__ == "__main__":
    # Dla Windows: multiprocessing wymaga tej funkcji
    import multiprocessing
    multiprocessing.freeze_support()
    
    run_experiment()

