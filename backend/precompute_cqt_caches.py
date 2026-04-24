import os
import glob
import multiprocessing
import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm
import sys

# Import config from project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config

def compute_cqt(file_path):
    cache_path = file_path + ".cqt_multi.pt"
    if os.path.exists(cache_path):
        return  # Already cached

    try:
        y_tensor, sr_load = torchaudio.load(file_path)
        y_tensor = y_tensor.mean(dim=0, keepdim=True) # Ensure mono
        
        if sr_load != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_load, new_freq=config.SAMPLE_RATE)
            y_tensor = resampler(y_tensor)
            
        y = y_tensor.squeeze(0).numpy()
        num_frames = int(np.ceil(len(y) / config.HOP_LENGTH))
        
        # HPSS Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        def process_waveform(waveform):
            cqt_spec = librosa.cqt(
                y=waveform, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
            )
            log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
            
            # Padding/Truncating to align exactly with num_frames based on hop length length expectations
            if log_cqt_spec.shape[1] > num_frames:
                log_cqt_spec = log_cqt_spec[:, :num_frames]
            elif log_cqt_spec.shape[1] < num_frames:
                pad_width = num_frames - log_cqt_spec.shape[1]
                log_cqt_spec = np.pad(log_cqt_spec, ((0, 0), (0, pad_width)), mode='constant')
            
            return log_cqt_spec

        cqt_orig = process_waveform(y)
        cqt_harm = process_waveform(y_harmonic)
        cqt_perc = process_waveform(y_percussive)
        
        # Stack to create [3, N_BINS, T] tensor
        stacked_features = np.stack([cqt_orig, cqt_harm, cqt_perc], axis=0)
        
        features = torch.tensor(stacked_features, dtype=torch.float32)
        torch.save(features, cache_path)
    except Exception as e:
        print(f"\nError processing {file_path}:\n{e}")

def main():
    dataset_dir = r"D:\Music\datasets"
    print(f"Scanning {dataset_dir} for audio files (Ultra Fast Mode)...", flush=True)

    flac_files = []
    wav_files = []
    scanned_count = 0
    scanned_dirs = 0

    for root, dirs, files in os.walk(dataset_dir):
        scanned_dirs += 1
        if scanned_dirs % 100 == 0:
            print(f"  [Directory Scan Progress] Currently at: {root} (Scanned {scanned_dirs} dirs, {scanned_count} files so far...)", flush=True)
            
        for file in files:
            scanned_count += 1
            if file.endswith("_nonoise_mono_body.flac"):
                flac_files.append(os.path.join(root, file))
            elif file.endswith("_mic.wav") and "GuitarSet" in root:
                wav_files.append(os.path.join(root, file))

    all_files = flac_files + wav_files
    print(f"\nScan complete! Checked {scanned_dirs} dirs and {scanned_count} files.", flush=True)
    print(f"Found {len(flac_files)} SynthTab files and {len(wav_files)} GuitarSet files. Total target: {len(all_files)}", flush=True)

    # Filter out files that already have caches
    pending_files = [f for f in all_files if not os.path.exists(f + ".cqt_multi.pt")]
    print(f"Files needing caching: {len(pending_files)}", flush=True)

    if not pending_files:
        print("All caches are already computed.", flush=True)
        return

    # Prevent Out-Of-Memory (Unable to allocate) errors by restricting workers
    # User requested 8 workers to balance speed and stability
    num_workers = 8
    print(f"Starting ProcessPoolExecutor with {num_workers} workers (Balanced Mode)...", flush=True)
    
    # ProcessPoolExecutor requires periodic memory clearing in heavy librosa tasks
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=10) as pool:
        # Use imap_unordered to see progress
        for _ in tqdm(pool.imap_unordered(compute_cqt, pending_files), total=len(pending_files), desc="Precomputing CQT"):
            pass
            
    print("Precomputation finished!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
