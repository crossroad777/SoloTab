import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from training import loss_functions, epoch_processing
from model import architecture
import config

# Import logic from train_multi_dataset
from train_multi_dataset import get_track_lists, RobustGuitarDataset

def run_dry_run_speed_test():
    print("=" * 60)
    print(" Starting Dry-Run Speed Validation (10 Batches) ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load small subset (200 tracks) to avoid long initialization
    ds_map = get_track_lists()
    all_tracks = ds_map["GuitarSet"] + ds_map["Synthetic"]
    if len(all_tracks) > 200:
        all_tracks = all_tracks[:200]
        
    dataset = RobustGuitarDataset(all_tracks, augment=False)
    
    # We only test 10 batches (approx 10 * 8 = 80 samples)
    if len(dataset) > 80:
        test_subset = Subset(dataset, range(80))
    else:
        test_subset = dataset
        
    loader = DataLoader(
        test_subset, batch_size=8,
        shuffle=False, collate_fn=collate_fn_pad, num_workers=0
    )

    # Initialize Dummy Model
    print("Initializing dummy CRNN model...")
    with torch.no_grad():
        out = architecture.TabCNN()(torch.randn(1, 3, config.N_BINS_CQT, 32))
        cnn_out_dim = out.shape[1] * out.shape[2]

    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type="CONFORMER",
        rnn_hidden_size=256,
        rnn_layers=1,
        rnn_dropout=0.1,
    )
    model.to(device)
    
    criterion = loss_functions.CombinedLoss(onset_loss_weight=1.0).to(device)

    print("\n--- Starting Evaluation Loop ---")
    start_time = time.time()
    
    # Run our updated evaluate_one_epoch (the one with our numpy frames_to_notes fix)
    class DummyPbar(list):
        def set_postfix(self, *args, **kwargs): pass
    
    pbar = DummyPbar(loader)
    
    val_metrics = epoch_processing.evaluate_one_epoch(model, pbar, criterion, device, config)
    
    end_time = time.time()
    
    print("\n[Dry-Run Validation Result]")
    print(f"Evaluation of 10 batches took: {end_time - start_time:.4f} seconds")
    print(f"Calculated Metrics Keys: {list(val_metrics.keys())}")
    print("=" * 60)
    
if __name__ == "__main__":
    run_dry_run_speed_test()
