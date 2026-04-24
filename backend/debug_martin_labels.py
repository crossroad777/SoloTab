import sys
import os
import torch # type: ignore

project_root = r"D:\Music\nextchord-solotab"
sys.path.insert(0, os.path.join(project_root, "backend"))
sys.path.insert(0, os.path.join(project_root, "music-transcription", "python"))

from resume_martin_training import MartinSynthDataset # type: ignore
import config # type: ignore

ds = MartinSynthDataset(r"D:\Music\datasets\martin_finger", augment=False)
if len(ds) > 0:
    for i in range(min(3, len(ds))):
        print(f"\n--- Sample {i} ---")
        features, (onset_labels, fret_labels), raw_labels, flac_path = ds[i]
        print(f"Features: {features.shape}")
        print(f"Onset Labels: Max={onset_labels.max().item()}, Sum={onset_labels.sum().item()}")
        
        silents = torch.sum(fret_labels == config.MAX_FRETS + 1).item()
        total_f = fret_labels.numel()
        print(f"Fret Labels: Total={total_f}, Silence={silents}, Non-Silence={total_f - silents}")
        print(f"Raw Labels: {raw_labels.shape}")
