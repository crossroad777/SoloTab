"""
technique_classifier.py — ギター奏法テクニック分類器
=====================================================
AG-PT-set + IDMT-SMTデータで学習し、音声セグメントから
演奏テクニックを分類するCNNモデル。

テクニッククラス:
  0: normal (通常ピッキング)
  1: palm_mute (パームミュート)
  2: harmonic (ナチュラルハーモニクス)
  3: bend (ベンド)
  4: slide (スライド)
  5: vibrato (ビブラート)
  6: dead_note (デッドノート/ゴーストノート)

Usage:
    python technique_classifier.py train [--epochs 50]
    python technique_classifier.py predict --audio path/to/audio.wav --onset 1.5
"""

import os
import sys
import argparse
import glob
import json
import time

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─── Constants ───
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_DURATION = 0.5  # seconds per note segment
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)

# Technique classes
TECHNIQUE_CLASSES = [
    'normal',        # 0: Standard picking
    'palm_mute',     # 1: Palm mute
    'harmonic',      # 2: Natural/artificial harmonics
    'bend',          # 3: Bend
    'slide',         # 4: Slide
    'vibrato',       # 5: Vibrato
    'dead_note',     # 6: Dead/ghost note
]
NUM_CLASSES = len(TECHNIQUE_CLASSES)

# IDMT expression -> class mapping
IDMT_EXPR_MAP = {
    'NO': 0,  # normal
    'PM': 1,  # palm mute
    'HA': 2,  # harmonic
    'BE': 3,  # bend
    'SL': 4,  # slide
    'VI': 5,  # vibrato
    'DE': 6,  # dead note
}

# AG-PT technique_id -> class mapping
AGPT_TECH_MAP = {
    4: 2,  # Natural Harmonics -> harmonic
    5: 1,  # Palm Mute -> palm_mute
    6: 0,  # Pick Near Bridge -> normal (playing style, not technique)
    7: 0,  # Pick Over Soundhole -> normal
}


# ─── Model ───
class TechniqueClassifierCNN(nn.Module):
    """
    Small CNN for guitar technique classification from Mel spectrograms.
    Input: [B, 1, N_MELS, time_frames]
    Output: [B, NUM_CLASSES]
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Dataset ───
def extract_mel_segment(audio_path, onset_sec, duration=SEGMENT_DURATION, sr=SAMPLE_RATE):
    """Extract a Mel spectrogram segment around a note onset."""
    try:
        start_sample = max(0, int(onset_sec * sr) - int(0.05 * sr))  # 50ms before onset
        end_sample = start_sample + SEGMENT_SAMPLES
        y, _ = librosa.load(audio_path, sr=sr, mono=True,
                             offset=start_sample/sr,
                             duration=SEGMENT_DURATION + 0.1)
        
        if len(y) < SEGMENT_SAMPLES // 2:
            return None
        
        # Pad if needed
        if len(y) < SEGMENT_SAMPLES:
            y = np.pad(y, (0, SEGMENT_SAMPLES - len(y)))
        else:
            y = y[:SEGMENT_SAMPLES]
        
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        return mel_db.astype(np.float32)
    except Exception:
        return None


class TechniqueDataset(Dataset):
    """Combined AG-PT-set + IDMT technique dataset."""
    
    def __init__(self, items, augment=False):
        """items: list of (mel_spectrogram, class_id)"""
        self.items = items
        self.augment = augment
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        mel, label = self.items[idx]
        # Add channel dim: [1, N_MELS, T]
        x = torch.FloatTensor(mel).unsqueeze(0)
        
        if self.augment:
            # Time masking
            if np.random.random() < 0.3:
                t = x.shape[-1]
                mask_len = np.random.randint(1, max(2, t // 4))
                mask_start = np.random.randint(0, max(1, t - mask_len))
                x[:, :, mask_start:mask_start+mask_len] = 0
            # Frequency masking
            if np.random.random() < 0.3:
                f = x.shape[-2]
                mask_len = np.random.randint(1, max(2, f // 8))
                mask_start = np.random.randint(0, max(1, f - mask_len))
                x[:, mask_start:mask_start+mask_len, :] = 0
        
        return x, torch.LongTensor([label])[0]


def load_agpt_data(data_root):
    """Load AG-PT-set data."""
    import pandas as pd
    
    metadata_dir = os.path.join(data_root, 'metadata')
    audio_dir = os.path.join(data_root, 'data', 'audio', 'audio')
    
    labels = pd.read_csv(os.path.join(metadata_dir, 'note_labels.csv'))
    
    items = []
    skipped = 0
    
    for _, row in labels.iterrows():
        tech_id = row['expressive_technique_id']
        
        # Map to our classes
        if tech_id not in AGPT_TECH_MAP:
            skipped += 1
            continue
        
        class_id = AGPT_TECH_MAP[tech_id]
        onset_sec = row['onset_label_seconds']
        audio_file = row['audio_file_path']
        # audio_file_path already has .wav extension
        audio_path = os.path.join(audio_dir, audio_file)
        if not os.path.exists(audio_path):
            # Try without extension
            audio_path = os.path.join(audio_dir, audio_file + '.wav')
            if not os.path.exists(audio_path):
                skipped += 1
                continue
        
        mel = extract_mel_segment(audio_path, onset_sec)
        if mel is not None:
            items.append((mel, class_id))
    
    return items, skipped


def load_idmt_data(idmt_root):
    """Load IDMT-SMT-Guitar single-note data with technique labels."""
    import xml.etree.ElementTree as ET
    
    items = []
    skipped = 0
    
    # dataset1 has single notes, dataset4 has more guitars
    for ds_name in ['dataset1', 'dataset4']:
        ds_path = os.path.join(idmt_root, ds_name)
        if not os.path.isdir(ds_path):
            continue
        
        # Walk all subdirs
        for root_dir, dirs, files in os.walk(ds_path):
            for f in files:
                if not f.endswith('.xml'):
                    continue
                
                xml_path = os.path.join(root_dir, f)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    transcription = root.find('transcription')
                    if transcription is None:
                        continue
                    
                    events = transcription.findall('event')
                    
                    # Get audio path
                    global_param = root.find('globalParameter')
                    audio_file = global_param.find('audioFileName').text.strip().lstrip('\\')
                    audio_dir = root_dir.replace('annotation', 'audio')
                    audio_path = os.path.join(audio_dir, audio_file)
                    
                    if not os.path.exists(audio_path):
                        continue
                    
                    for event in events:
                        expr = event.find('expressionStyle')
                        if expr is None:
                            continue
                        expr_style = expr.text.strip()
                        
                        if expr_style not in IDMT_EXPR_MAP:
                            continue
                        
                        class_id = IDMT_EXPR_MAP[expr_style]
                        onset = float(event.find('onsetSec').text)
                        
                        mel = extract_mel_segment(audio_path, onset)
                        if mel is not None:
                            items.append((mel, class_id))
                
                except Exception:
                    skipped += 1
    
    return items, skipped


def train_model(args):
    """Train the technique classifier."""
    print("=" * 60)
    print("Guitar Technique Classifier Training")
    print("=" * 60)
    
    # Load AG-PT-set
    agpt_root = os.path.join('..', 'datasets', 'AG-PT-set', 'aGPTset')
    idmt_root = os.path.join('..', 'datasets', 'IDMT-SMT-GUITAR_V2', 'IDMT-SMT-GUITAR_V2')
    
    all_items = []
    
    print("\nLoading AG-PT-set data...")
    agpt_items, agpt_skipped = load_agpt_data(agpt_root)
    print(f"  AG-PT-set: {len(agpt_items)} items loaded, {agpt_skipped} skipped")
    all_items.extend(agpt_items)
    
    print("\nLoading IDMT-SMT data...")
    idmt_items, idmt_skipped = load_idmt_data(idmt_root)
    print(f"  IDMT-SMT: {len(idmt_items)} items loaded, {idmt_skipped} skipped")
    all_items.extend(idmt_items)
    
    print(f"\nTotal: {len(all_items)} items")
    
    # Class distribution
    class_counts = {}
    for _, label in all_items:
        class_counts[label] = class_counts.get(label, 0) + 1
    print("\nClass distribution:")
    for cid in range(NUM_CLASSES):
        count = class_counts.get(cid, 0)
        print(f"  {cid}: {TECHNIQUE_CLASSES[cid]:15s} n={count}")
    
    if len(all_items) < 100:
        print("ERROR: Not enough data!")
        return
    
    # Train/val split (80/20)
    np.random.seed(42)
    np.random.shuffle(all_items)
    split = int(len(all_items) * 0.8)
    train_items = all_items[:split]
    val_items = all_items[split:]
    
    train_dataset = TechniqueDataset(train_items, augment=True)
    val_dataset = TechniqueDataset(val_items, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = TechniqueClassifierCNN(NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Class weights for imbalanced data
    total = sum(class_counts.values())
    weights = torch.FloatTensor([total / max(class_counts.get(i, 1), 1) for i in range(NUM_CLASSES)])
    weights = weights / weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Output
    output_dir = os.path.join('..', 'generated', 'technique_classifier')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining: {args.epochs} epochs, LR={args.lr}, BS={args.batch_size}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}\n")
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        train_acc = train_correct / max(train_total, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  [{epoch+1}/{args.epochs}] "
                  f"train_loss={train_loss/len(train_loader):.4f} "
                  f"train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f} "
                  f"({elapsed:.0f}s)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': TECHNIQUE_CLASSES,
                'val_acc': val_acc,
                'epoch': epoch + 1,
            }, best_path)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Model: {best_path}")
    
    # Save class info
    info = {
        'classes': TECHNIQUE_CLASSES,
        'best_val_acc': best_val_acc,
        'n_train': len(train_items),
        'n_val': len(val_items),
        'class_counts': {TECHNIQUE_CLASSES[k]: v for k, v in class_counts.items()},
    }
    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def predict_technique(audio_path, onset_sec, model_path=None):
    """Predict technique for a single note."""
    if model_path is None:
        model_path = os.path.join('..', 'generated', 'technique_classifier', 'best_model.pt')
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = TechniqueClassifierCNN(len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    mel = extract_mel_segment(audio_path, onset_sec)
    if mel is None:
        return None, 0.0
    
    x = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
    
    return checkpoint['classes'][pred_class], confidence


def main():
    parser = argparse.ArgumentParser(description='Guitar Technique Classifier')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    
    # Predict
    pred_parser = subparsers.add_parser('predict')
    pred_parser.add_argument('--audio', required=True)
    pred_parser.add_argument('--onset', type=float, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        tech, conf = predict_technique(args.audio, args.onset)
        print(f"Technique: {tech} (confidence: {conf:.2f})")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
