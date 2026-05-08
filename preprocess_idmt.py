"""
preprocess_idmt.py — IDMT-SMT-Guitar V2 を学習パイプライン互換形式に変換
======================================================================
Dataset1 (400 pairs) + Dataset2 (261 pairs) → features.pt / labels.pt
"""
import os, sys, glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
import librosa
from tqdm import tqdm

sys.path.insert(0, r"d:\Music\nextchord-solotab\music-transcription\python")
import config

IDMT_ROOT = r"D:\Music\Datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2"
OUTPUT_DIR = r"D:\Music\datasets\idmt_processed"
SR = config.SAMPLE_RATE  # 22050


def parse_xml(xml_path):
    """Parse IDMT XML annotation → list of (onset, offset, string, fret, pitch)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []
    for ev in root.findall(".//event"):
        onset = float(ev.findtext("onsetSec", "0"))
        offset = float(ev.findtext("offsetSec", "0"))
        pitch = int(ev.findtext("pitch", "0"))
        fret = int(ev.findtext("fretNumber", "0"))
        string_num = int(ev.findtext("stringNumber", "0"))
        # IDMT: string 1-6 (1=low E) → pipeline: 0-5
        string_idx = string_num - 1
        if 0 <= string_idx <= 5 and pitch > 0 and onset < offset:
            events.append([onset, offset, float(string_idx), float(fret), float(pitch)])
    return events


def compute_cqt(audio_path):
    """Load audio and compute CQT features matching pipeline."""
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    cqt = librosa.cqt(
        y=audio, sr=SR,
        hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT,
        n_bins=config.N_BINS_CQT,
        bins_per_octave=config.BINS_PER_OCTAVE_CQT,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return torch.tensor(log_cqt, dtype=torch.float32)


def find_pairs(dataset_dir):
    """Find (xml_path, wav_path) pairs in a dataset directory."""
    pairs = []
    
    # Dataset1 style: annotation/XXX.xml, audio/XXX.wav in same subdirectory
    for subdir in glob.glob(os.path.join(dataset_dir, "*")):
        if not os.path.isdir(subdir):
            continue
        ann_dir = os.path.join(subdir, "annotation")
        aud_dir = os.path.join(subdir, "audio")
        if os.path.isdir(ann_dir) and os.path.isdir(aud_dir):
            for xml_file in glob.glob(os.path.join(ann_dir, "*.xml")):
                base = os.path.splitext(os.path.basename(xml_file))[0]
                wav_file = os.path.join(aud_dir, base + ".wav")
                if os.path.exists(wav_file):
                    pairs.append((xml_file, wav_file))
    
    # Dataset2 style: annotation/ and audio/ at top level
    ann_dir = os.path.join(dataset_dir, "annotation")
    aud_dir = os.path.join(dataset_dir, "audio")
    if os.path.isdir(ann_dir) and os.path.isdir(aud_dir):
        for xml_file in glob.glob(os.path.join(ann_dir, "*.xml")):
            base = os.path.splitext(os.path.basename(xml_file))[0]
            wav_file = os.path.join(aud_dir, base + ".wav")
            if os.path.exists(wav_file):
                pairs.append((xml_file, wav_file))
    
    return pairs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect all pairs from dataset1 + dataset2
    all_pairs = []
    for ds_name in ["dataset1", "dataset2"]:
        ds_dir = os.path.join(IDMT_ROOT, ds_name)
        pairs = find_pairs(ds_dir)
        print(f"  {ds_name}: {len(pairs)} pairs found")
        all_pairs.extend(pairs)
    
    print(f"\n  Total pairs: {len(all_pairs)}")
    
    ids = []
    errors = 0
    total_notes = 0
    
    for i, (xml_path, wav_path) in enumerate(tqdm(all_pairs, desc="Processing")):
        try:
            events = parse_xml(xml_path)
            if not events:
                errors += 1
                continue
            
            features = compute_cqt(wav_path)
            labels = torch.tensor(events, dtype=torch.float32)
            
            tid = f"idmt_{i:05d}"
            torch.save(features, os.path.join(OUTPUT_DIR, f"{tid}_features.pt"))
            torch.save(labels, os.path.join(OUTPUT_DIR, f"{tid}_labels.pt"))
            ids.append(tid)
            total_notes += len(events)
            
        except Exception as e:
            print(f"\n  Error [{i}] {os.path.basename(wav_path)}: {e}")
            errors += 1
    
    # Write train_ids.txt
    ids_path = os.path.join(OUTPUT_DIR, "train_ids.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(ids))
    
    print(f"\n{'='*60}")
    print(f"  IDMT-SMT Preprocessing Complete")
    print(f"{'='*60}")
    print(f"  Processed: {len(ids)} / {len(all_pairs)} (errors: {errors})")
    print(f"  Notes:     {total_notes}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  IDs file:  {ids_path}")


if __name__ == "__main__":
    main()
