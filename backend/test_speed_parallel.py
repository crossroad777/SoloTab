import time
import torch
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config
from train_multi_dataset import RobustGuitarDataset
from torch.utils.data import DataLoader
from data_processing.batching import collate_fn_pad

def test_speed():
    # Find a real file to test with
    # Create fake tracks pointing to it
    dummy_tracks = []
    
    # Check if a specific cache file exists
    sample_file = r"D:\Music\datasets\GuitarSet\audio_mono-mic\00_BN1-129-Eb_comp_mic.wav"
    if not os.path.exists(sample_file + ".cqt_multi.pt"):
        print("Could not find dummy file on D drive. Creating a mock CQT cache for test...")
        # Create a mock cache
        os.makedirs(r"D:\Music\datasets\GuitarSet\audio_mono-mic", exist_ok=True)
        torch.save(torch.randn(3, 192, 430), sample_file + ".cqt_multi.pt")
        # create empty jams too
        jams_dir = r"D:\Music\datasets\GuitarSet\annotation"
        os.makedirs(jams_dir, exist_ok=True)
        open(os.path.join(jams_dir, "00_BN1-129-Eb_comp.jams"), "w").close()

    jams_file = r"D:\Music\datasets\GuitarSet\annotation\00_BN1-129-Eb_comp.jams"
    
    for _ in range(300): # 300 instances of the same track
        dummy_tracks.append({"audio": sample_file, "jams": jams_file})

    dataset = RobustGuitarDataset(dummy_tracks, augment=True)
    
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=True, collate_fn=collate_fn_pad, 
        num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True
    )

    print("\n--- Testing DataLoader Speed ---")
    start_time = time.time()
    
    batch_count = 0
    for i, batch in enumerate(loader):
        batch_count += 1
        if i == 0:
            print(f"First batch load time: {time.time() - start_time:.4f} s")
        if i >= 10: # test 10 batches
            break
            
    total_time = time.time() - start_time
    print(f"Time for 10 batches: {total_time:.4f} s")
    if batch_count > 1:
        print(f"Average time per batch (excluding 1st start overhead): {(total_time - (time.time() - start_time))/9:.4f} s")
    
if __name__ == "__main__":
    test_speed()
