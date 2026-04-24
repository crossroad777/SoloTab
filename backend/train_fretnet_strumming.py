import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Fix CP932 console encoding issues on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')

from amt_tools.datasets import GuitarSet
from amt_tools.features import HCQT
from fretnet_augmentation import HCQTAugmenter, AugmentedDatasetWrapper
from amt_tools.train import train
from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedOffsetsWrapper, StackedNoteTranscriber
from amt_tools.evaluate import ComboEvaluator, TablatureEvaluator

def main():
    print("==================================================")
    print("FRETNET EXTREME FINE-TUNING FOR STRUMMING DATA")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. HCQT Features config
    data_proc = HCQT(sample_rate=22050, hop_length=512, fmin=73.42, 
                     harmonics=[0.5, 1, 2, 3, 4, 5], n_bins=144, bins_per_octave=36)
                     
    # 2. Model Loading (Load first to get the correct instrument profile)
    best_path = r"D:\Music\nextchord-solotab\generated\baseline_model_finetune\run_2\best_model.pth"
    if not os.path.exists(best_path):
        best_path = r"D:\Music\nextchord-solotab\backend\fretnet_models\models\fold-0\model-2500.pt"

    model = torch.load(best_path, map_location=device, weights_only=False)
    if hasattr(model, 'change_device'):
        gpu_id = 0 if torch.cuda.is_available() else -1
        model.change_device(gpu_id)
        
    print(f"Loaded anchor model from: {best_path}")

    # 3. Dataset Setup
    base_dir = r"D:\Music\nextchord-solotab\datasets\GuitarSet"
    
    # Custom splits targeting only SingerSongwriter tracks across all players
    splits = ['0', '1', '2', '3', '4', '5']
        
    print(f"Targeting splits: {splits}")
    
    # Load dataset using the model's built-in profile dynamically
    gset = GuitarSet(base_dir=base_dir, splits=splits, data_proc=data_proc, 
                     profile=model.profile, save_data=True, save_loc=r"D:\Music\nextchord-solotab\backend\fretnet_models\data_cache_ss")
    
    # Filter only for SingerSongwriter tracks! (And limit to first 3 tracks for immediate testing)
    filtered_tracks = [t for t in gset.tracks if 'SS' in t][:3]
    gset.tracks = filtered_tracks
    print(f"Filtered down to {len(gset.tracks)} SS tracks for strumming tuning")
                     
    # 3. Aggressive Augmentation Wrap (SAFETY ROLLBACK: minimal settings to prevent mode collapse)
    augmenter = HCQTAugmenter(
        enable=True,
        specaug_enabled=True,
        specaug_p=0.3,       # Reduced from 0.8
        freq_shift_enabled=False, # Disabled pitch shifting
        freq_shift_p=0.0,
        noise_enabled=False,   # Disabled extreme noise
        noise_p=0.0,
        gain_enabled=True,
        gain_p=0.3           # Reduced from 0.5
    )
    
    aug_dataset = AugmentedDatasetWrapper(gset, augmenter)
    
    # Freeze CNN layers to maintain core feature extraction robust, tune BLSTM
    for name, param in model.named_parameters():
        if "cnn" in name:
            param.requires_grad = False
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    
    # 5. Validation setup
    estimator = ComboEstimator([
        TablatureWrapper(profile=model.profile),
        StackedOffsetsWrapper(profile=model.profile),
        StackedNoteTranscriber(profile=model.profile)
    ])
    
    evaluator = ComboEvaluator([TablatureEvaluator(profile=model.profile)])
    
    log_dir = r"D:\Music\nextchord-solotab\generated\fretnet_models\models\strumming_finetuned"
    os.makedirs(log_dir, exist_ok=True)
    
    train_loader = DataLoader(aug_dataset, batch_size=1, shuffle=True, drop_last=False)
    
    print("\nStarting rapid safety fine-tuning...")
    # 6. Training Execution (50 loops/iterations, 5 checkpoints)
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        iterations=50,
        checkpoints=5,
        log_dir=log_dir,
        val_set=gset,
        estimator=estimator,
        evaluator=evaluator
    )

if __name__ == '__main__':
    main()
