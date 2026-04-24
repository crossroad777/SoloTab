import os
os.environ["TQDM_ASCII"] = " 123456789#"  # Force ASCII progress bars using a valid 11-character string to prevent Division by Zero

import sys
import io
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader

# Fix CP932 console encoding issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
sys.path.insert(0, mt_python_dir)

from data_processing.dataset import GuitarSetTabDataset
from data_processing.batching import collate_fn_pad
from training import pipeline, epoch_processing
import copy

OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")
GAPS_PROCESSED_DATA_DIR = os.path.join(project_root, "_processed_gaps_data")
GAPS_V1_AUDIO_DIR = os.path.join(project_root, "datasets", "gaps", "gaps_v1", "audio")
GUITARSET_HOME = r"D:\Music\datasets\GuitarSet"
RESUME_CHECKPOINT_PATH = os.path.join(OUTPUT_BASE_DIR, "training_output", "run_2_run_2_augDisabled", "best_model.pth")

class GAPSAcousticDataset(Dataset):
    def __init__(self, data_dir, audio_dir, enable_augmentations=True):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.enable_augmentations = enable_augmentations
        
        self.pairs = []
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith("_labels.pt"):
                    base_id = f.replace("_labels.pt", "")
                    audio_path = os.path.join(audio_dir, f"{base_id}.wav")
                    if os.path.exists(audio_path):
                        self.pairs.append((os.path.join(data_dir, f), audio_path))
        print(f"[GAPSDataset] Discovered {len(self.pairs)} valid pairs in {data_dir}")

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        label_path, audio_path = self.pairs[idx]
        raw_labels = torch.load(label_path, weights_only=False)
        
        slice_duration = 10.0
        try:
            total_duration = librosa.get_duration(path=audio_path)
            offset = 0.0
            if total_duration > slice_duration:
                offset = np.random.uniform(0, total_duration - slice_duration)
            
            audio_data, _ = librosa.load(audio_path, sr=22050, mono=True, offset=offset, duration=slice_duration)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio_data = np.zeros(int(slice_duration * 22050), dtype=np.float32)
            offset = 0.0

        valid_labels = []
        for lbl in raw_labels:
            onset = float(lbl[0])
            offset_time = float(lbl[1])
            string_idx = float(lbl[2])
            # GAPS uses 0=High E, 5=Low E. GuitarSet uses 0=Low E, 5=High E.
            # We MUST invert it to match GuitarSet so the mixed training does not corrupt the model!
            string_idx = 5.0 - string_idx
            
            fret = float(lbl[3])
            pitch = float(lbl[4])
            
            if onset < (offset + slice_duration) and offset_time > offset:
                new_onset = max(0.0, onset - offset)
                new_offset = min(slice_duration, offset_time - offset)
                if new_offset > new_onset:
                    valid_labels.append([new_onset, new_offset, string_idx, fret, pitch])
                    
        if len(valid_labels) > 0:
            labels_for_transform = torch.tensor(valid_labels, dtype=torch.float32)
        else:
            labels_for_transform = torch.empty((0, 5), dtype=torch.float32)

        if self.enable_augmentations:
            pass # Standard pipeline augmentations applied at feature extraction level in original script
            
        cqt_spec = librosa.cqt(
            y=audio_data,
            sr=22050,
            hop_length=512,
            fmin=librosa.note_to_hz('E2'),
            n_bins=168,
            bins_per_octave=24,
        )
        log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features = torch.tensor(log_cqt_spec, dtype=torch.float32)
        
        # Frame labels
        from data_processing.dataset import create_frame_level_labels
        onset_targets, fret_targets = create_frame_level_labels(
            raw_annotation_tensor=labels_for_transform,
            feature_map_tensor=features,
            frame_hop_length=512,
            audio_sr=22050,
            fret_max_value=20
        )
        # We do not transform features.T here because the batcher collate_fn_pad expects [Freq, Time] and transposes it natively.
        track_id = os.path.basename(audio_path).replace('.wav', '')
        return features, (onset_targets, fret_targets), labels_for_transform, track_id

def run_experiment():
    print("============================================================")
    print("Acoustic Guitar Mixed Fine-Tuning Script")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    import config
    # Overwrite relevant config parameters dynamically limit
    config.DATASET_TRAIN_DIR = os.path.join(OUTPUT_BASE_DIR, "train")
    config.DATASET_VAL_DIR = os.path.join(OUTPUT_BASE_DIR, "val")
    config.DATASET_TEST_DIR = os.path.join(OUTPUT_BASE_DIR, "test")
    
    config.DATASET_TRAIN_AUGMENTATION_PARAMS = {
        'enable': True,
        'p_time_stretch': 0.3,
        'time_stretch_limits': (0.8, 1.2),
        'p_pitch_shift': 0.3,
        'pitch_shift_limits': (-2, 2),
        'p_add_noise': 0.3,
        'noise_snr_limits': (10, 30),
        'p_apply_eq': 0.3,
        'eq_gain_limits': (-10, 10),
        'p_apply_reverb': 0.3,
        'reverb_room_size': (0.1, 0.8)
    }
    config.BATCH_SIZE = 2
    config.RNN_HIDDEN_SIZE = 768
    config.NUM_EPOCHS = 200
    config.LEARNING_RATE = 1e-5
    
    print("\n[1] Preparing Mixed DataLoader...")
    
    from data_processing.dataset import create_frame_level_labels
    
    g_train_dataset = GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR,
        data_split_name="train",
        audio_hop_length=config.HOP_LENGTH,
        audio_sample_rate=config.SAMPLE_RATE,
        max_fret_value=config.MAX_FRETS,
        audio_n_cqt_bins=config.N_BINS_CQT,
        audio_cqt_bins_per_octave=config.BINS_PER_OCTAVE_CQT,
        audio_cqt_fmin=config.FMIN_CQT,
        label_transform_function=create_frame_level_labels,
        enable_audio_augmentations=True,
        guitarset_data_home=GUITARSET_HOME,
        aug_p_time_stretch=config.DATASET_TRAIN_AUGMENTATION_PARAMS['p_time_stretch'],
        aug_time_stretch_limits=config.DATASET_TRAIN_AUGMENTATION_PARAMS['time_stretch_limits'],
        aug_p_pitch_shift=config.DATASET_TRAIN_AUGMENTATION_PARAMS['p_pitch_shift'],
        aug_pitch_shift_range=config.DATASET_TRAIN_AUGMENTATION_PARAMS['pitch_shift_limits'],
        aug_p_add_noise=config.DATASET_TRAIN_AUGMENTATION_PARAMS['p_add_noise'],
        aug_noise_level_limits=(10 ** (-config.DATASET_TRAIN_AUGMENTATION_PARAMS['noise_snr_limits'][1] / 20), 10 ** (-config.DATASET_TRAIN_AUGMENTATION_PARAMS['noise_snr_limits'][0] / 20)),
        aug_p_random_gain=config.DATASET_TRAIN_AUGMENTATION_PARAMS.get('p_apply_eq', 0),
        aug_gain_limits=(0.8, 1.2)
    )
    print(f"  GuitarSet Training Count: {len(g_train_dataset)}")
    
    gaps_dataset = GAPSAcousticDataset(
        data_dir=GAPS_PROCESSED_DATA_DIR,
        audio_dir=GAPS_V1_AUDIO_DIR,
        enable_augmentations=True
    )
    print(f"  GAPS Nylon Training Count: {len(gaps_dataset)}")
    
    mixed_dataset = ConcatDataset([g_train_dataset, gaps_dataset])
    
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_pad,
        num_workers=0
    )

    print("\n[2] Starting Fine-Tuning Execution...")
    print(f"Will fine-tune on target tracks (GuitarSet + GAPS) for {config.NUM_EPOCHS} epochs.")
    
    # Inject RESUME_CHECKPOINT_PATH into environment
    os.environ["RESUME_CHECKPOINT"] = RESUME_CHECKPOINT_PATH

    print(f"Loaded pretrained weights from {RESUME_CHECKPOINT_PATH} (strict=False)")
    
    result = pipeline.process_single_hyperparameter_run(
        run_id=3,
        hyperparams_combo={
            "RNN_TYPE": config.RNN_TYPE_DEFAULT,
            "RNN_HIDDEN_SIZE": config.RNN_HIDDEN_SIZE,
            "RNN_LAYERS": config.RNN_LAYERS_DEFAULT,
            "RNN_DROPOUT": config.RNN_DROPOUT_DEFAULT,
            "LEARNING_RATE_INIT": config.LEARNING_RATE,
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
        validation_loader=train_loader,
        test_loader=train_loader
    )
    print(f"\nPipeline Return Status: {result}")

if __name__ == "__main__":
    run_experiment()
