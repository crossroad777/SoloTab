"""
preprocess_idmt.py — IDMT-SMT-GUITAR V2 前処理
================================================================
XML注釈 + WAV → CQT features.pt + raw_labels.pt

Usage:
    python preprocess_idmt.py
"""
import os, sys, glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
import librosa

mt_python_dir = r"D:\Music\nextchord-solotab\music-transcription\python"
sys.path.insert(0, mt_python_dir)
import config

IDMT_BASE = r"D:\Music\Datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2\dataset2"
AUDIO_DIR = os.path.join(IDMT_BASE, "audio")
ANNO_DIR = os.path.join(IDMT_BASE, "annotation")
OUTPUT_DIR = r"D:\Music\datasets\idmt_processed"

# Standard tuning MIDI
OPEN_STRING_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}


def parse_idmt_xml(xml_path):
    """
    IDMT XML → raw_labels [N, 5] (onset, offset, string, fret, pitch)
    XML has: onsetSec, offsetSec, pitch, stringNumber, fretNumber
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    notes = []
    for event in root.findall('.//event'):
        onset = float(event.findtext('onsetSec', '0'))
        offset = float(event.findtext('offsetSec', '0'))
        pitch = int(event.findtext('pitch', '0'))
        string_num = int(event.findtext('stringNumber', '0'))  # 1-indexed in XML
        fret = int(event.findtext('fretNumber', '0'))
        
        # IDMT uses 1-indexed strings (1=low E, 6=high E)
        # Our system uses 0-indexed (0=low E, 5=high E)
        string_idx = string_num - 1
        
        if string_idx < 0 or string_idx > 5:
            continue
        if fret < 0 or fret > 19:
            continue
        if pitch < 40 or pitch > 88:
            continue
            
        notes.append([onset, offset, float(string_idx), float(fret), float(pitch)])
    
    if not notes:
        return torch.zeros((0, 5), dtype=torch.float32)
    
    labels = np.array(notes, dtype=np.float32)
    labels = labels[labels[:, 0].argsort()]
    return torch.from_numpy(labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    xml_files = sorted(glob.glob(os.path.join(ANNO_DIR, "*.xml")))
    print(f"Found {len(xml_files)} XML annotations")
    
    processed = []
    skipped = 0
    
    for i, xml_path in enumerate(xml_files):
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        wav_path = os.path.join(AUDIO_DIR, basename + ".wav")
        
        if not os.path.exists(wav_path):
            skipped += 1
            continue
        
        # Parse labels
        raw_labels = parse_idmt_xml(xml_path)
        if len(raw_labels) == 0:
            skipped += 1
            continue
        
        # Extract CQT features
        try:
            y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
            cqt_spec = librosa.cqt(
                y=y, sr=sr, hop_length=config.HOP_LENGTH,
                fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
                bins_per_octave=config.BINS_PER_OCTAVE_CQT
            )
            log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
            features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"  [{i+1}] ERROR {basename}: {e}")
            skipped += 1
            continue
        
        # Save
        feat_path = os.path.join(OUTPUT_DIR, f"{basename}_features.pt")
        label_path = os.path.join(OUTPUT_DIR, f"{basename}_labels.pt")
        torch.save(features, feat_path)
        torch.save(raw_labels, label_path)
        processed.append(basename)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(xml_files):
            print(f"  [{i+1}/{len(xml_files)}] Processed: {len(processed)}, Skipped: {skipped}")
    
    # Save train_ids.txt
    ids_path = os.path.join(OUTPUT_DIR, "train_ids.txt")
    with open(ids_path, 'w') as f:
        for tid in processed:
            f.write(tid + "\n")
    
    print(f"\n=== IDMT-SMT-V2 Preprocessing Complete ===")
    print(f"  Processed: {len(processed)}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  IDs file: {ids_path}")


if __name__ == "__main__":
    main()
