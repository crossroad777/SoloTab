"""CQT spectrogram comparison between GuitarSet, FluidR3_GM, and Pianoteq8"""
import sys
sys.path.insert(0, r"d:\Music\nextchord-solotab\music-transcription\python")
import torch
import numpy as np

# Paths
gs_path = r"d:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\train\00_BN1-129-Eb_solo_features.pt"
fluid_path = r"D:\Music\datasets\synth_v2\synth_v2_00000_features.pt"
piano_path = r"D:\Music\datasets\synth_pianoteq_test\synth_v2_00000_features.pt"

gs = torch.load(gs_path, weights_only=False).numpy()
fluid = torch.load(fluid_path, weights_only=False).numpy()
piano = torch.load(piano_path, weights_only=False).numpy()

print("=== CQT Feature Statistics ===")
for name, feat in [("GuitarSet", gs), ("FluidR3_GM", fluid), ("Pianoteq8", piano)]:
    print(f"  {name:12s}: shape={feat.shape}, mean={feat.mean():.2f}, std={feat.std():.2f}, min={feat.min():.2f}, max={feat.max():.2f}")

print()
print("=== Energy per octave (mean dB) ===")
bpo = 36  # bins_per_octave
header = "  " + " " * 14 + " | ".join([f"Oct{o}" for o in range(6)])
print(header)
for name, feat in [("GuitarSet", gs), ("FluidR3_GM", fluid), ("Pianoteq8", piano)]:
    n_bins = feat.shape[0]
    octaves = n_bins // bpo
    row = []
    for o in range(min(octaves, 6)):
        start = o * bpo
        end = start + bpo
        row.append(f"{feat[start:end, :].mean():6.1f}")
    print(f"  {name:12s}: {' | '.join(row)}")

print()
print("=== Spectral Centroid (weighted bin index) ===")
for name, feat in [("GuitarSet", gs), ("FluidR3_GM", fluid), ("Pianoteq8", piano)]:
    energy = feat - feat.min()
    weighted_sum = np.sum(energy * np.arange(feat.shape[0])[:, None], axis=0)
    total = np.sum(energy, axis=0) + 1e-8
    centroid = weighted_sum / total
    print(f"  {name:12s}: mean_centroid={centroid.mean():.1f}, std={centroid.std():.1f}")

print()
print("=== Spectral Flatness (silence ratio) ===")
for name, feat in [("GuitarSet", gs), ("FluidR3_GM", fluid), ("Pianoteq8", piano)]:
    silence = (feat < -70).mean()
    print(f"  {name:12s}: {silence*100:.1f}% bins < -70dB")
