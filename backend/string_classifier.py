"""
string_classifier.py — 弦推定CNN
=================================
GuitarSet hex-debleeded (弦別分離音声) + JAMSアノテーションから
ノートの弦を音色で推定するCNNモデル。

入力: ノートのMelスペクトログラム (ミックス音声から抽出)
出力: 6弦のどの弦か (0-5, 0=6弦E, 5=1弦E)

Usage:
    python string_classifier.py train [--epochs 30]
    python string_classifier.py evaluate
"""

import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import librosa
import soundfile as sf
import jams
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─── Constants ───
SAMPLE_RATE = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
SEGMENT_DURATION = 0.3  # seconds per note
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)
NUM_STRINGS = 6

GUITARSET_JAMS_DIR = os.path.join('..', 'datasets', 'guitarset_annotation')
GUITARSET_HEX_DIR = os.path.join('..', 'datasets', 'guitarset_hex_debleeded')


# ─── Model ───
class StringClassifierCNN(nn.Module):
    """CNN for guitar string classification from Mel spectrograms."""
    def __init__(self, num_strings=NUM_STRINGS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 1, 128),  # +1 for MIDI pitch input
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_strings),
        )

    def forward(self, mel, pitch):
        """
        mel: [B, 1, N_MELS, T]
        pitch: [B, 1] normalized MIDI pitch
        """
        x = self.features(mel)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, pitch], dim=1)
        x = self.classifier(x)
        return x


# ─── Data Loading ───
def load_guitarset_notes():
    """
    Load note annotations from GuitarSet JAMS + hex audio.
    Returns list of (mel_spectrogram, midi_pitch, string_idx) tuples.
    """
    items = []
    skipped = 0

    jams_files = sorted(glob.glob(os.path.join(GUITARSET_JAMS_DIR, '*.jams')))
    if not jams_files:
        print(f"No JAMS files in {GUITARSET_JAMS_DIR}")
        return items, skipped

    print(f"  Found {len(jams_files)} JAMS files")

    for jams_path in jams_files:
        basename = os.path.splitext(os.path.basename(jams_path))[0]

        # Find matching hex audio
        # JAMS: 00_BN1-129-Eb_solo.jams -> hex: 00_BN1-129-Eb_solo_hex_cln.wav
        hex_path = os.path.join(GUITARSET_HEX_DIR, f'{basename}_hex_cln.wav')
        if not os.path.exists(hex_path):
            skipped += 1
            continue

        try:
            # Load 6-channel hex audio
            hex_data, hex_sr = sf.read(hex_path)
            if hex_data.ndim == 1:
                skipped += 1
                continue

            # Mix to mono for input features
            mono = np.mean(hex_data, axis=1)

            # Resample if needed
            if hex_sr != SAMPLE_RATE:
                mono = librosa.resample(mono, orig_sr=hex_sr, target_sr=SAMPLE_RATE)

            jam = jams.load(jams_path)

            # Extract notes per string
            for ann in jam.annotations:
                if ann.namespace != 'note_midi':
                    continue

                src = ann.annotation_metadata.data_source
                if src is None:
                    continue

                try:
                    string_idx = int(src)  # 0=lowest string (6th)
                except ValueError:
                    continue

                if string_idx < 0 or string_idx >= 6:
                    continue

                for obs in ann.data:
                    onset_sec = obs.time
                    midi_pitch = obs.value

                    # Extract mel segment from mono mix
                    start_sample = max(0, int(onset_sec * SAMPLE_RATE) - int(0.02 * SAMPLE_RATE))
                    end_sample = start_sample + SEGMENT_SAMPLES

                    if end_sample > len(mono):
                        continue

                    segment = mono[start_sample:end_sample]

                    if len(segment) < SEGMENT_SAMPLES // 2:
                        continue

                    if len(segment) < SEGMENT_SAMPLES:
                        segment = np.pad(segment, (0, SEGMENT_SAMPLES - len(segment)))

                    # Compute mel
                    mel = librosa.feature.melspectrogram(
                        y=segment.astype(np.float32), sr=SAMPLE_RATE,
                        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
                    )
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

                    items.append((mel_db.astype(np.float32), float(midi_pitch), string_idx))

        except Exception as e:
            skipped += 1

    return items, skipped


class StringDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mel, pitch, string_idx = self.items[idx]
        x = torch.FloatTensor(mel).unsqueeze(0)  # [1, N_MELS, T]

        # Normalize pitch to [0, 1] range (MIDI 30-90)
        pitch_norm = (pitch - 30.0) / 60.0
        pitch_tensor = torch.FloatTensor([pitch_norm])

        if self.augment:
            # Time masking
            if np.random.random() < 0.3:
                t = x.shape[-1]
                mask_len = np.random.randint(1, max(2, t // 4))
                mask_start = np.random.randint(0, max(1, t - mask_len))
                x[:, :, mask_start:mask_start + mask_len] = 0

        return x, pitch_tensor, torch.LongTensor([string_idx])[0]


def train_model(args):
    print("=" * 60)
    print("Guitar String Classifier Training")
    print("=" * 60)

    print("\nLoading GuitarSet hex data...")
    items, skipped = load_guitarset_notes()
    print(f"  Loaded: {len(items)} notes, skipped: {skipped}")

    if len(items) < 100:
        print("Not enough data!")
        return

    # Class distribution
    class_counts = {}
    for _, _, s in items:
        class_counts[s] = class_counts.get(s, 0) + 1
    print("\nString distribution:")
    string_names = ['6th (E2)', '5th (A2)', '4th (D3)', '3rd (G3)', '2nd (B3)', '1st (E4)']
    for i in range(NUM_STRINGS):
        print(f"  {i}: {string_names[i]:12s} n={class_counts.get(i, 0)}")

    # Split
    np.random.seed(42)
    np.random.shuffle(items)
    split = int(len(items) * 0.8)
    train_items = items[:split]
    val_items = items[split:]

    train_dataset = StringDataset(train_items, augment=True)
    val_dataset = StringDataset(val_items, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = StringClassifierCNN(NUM_STRINGS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Class weights
    total = sum(class_counts.values())
    weights = torch.FloatTensor([total / max(class_counts.get(i, 1), 1) for i in range(NUM_STRINGS)])
    weights = weights / weights.sum() * NUM_STRINGS
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    output_dir = os.path.join('..', 'generated', 'string_classifier')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining: {args.epochs} epochs, LR={args.lr}, BS={args.batch_size}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}\n")

    best_val_acc = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for mel, pitch, label in train_loader:
            mel, pitch, label = mel.to(device), pitch.to(device), label.to(device)
            out = model(mel, pitch)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == label).sum().item()
            train_total += label.size(0)

        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for mel, pitch, label in val_loader:
                mel, pitch, label = mel.to(device), pitch.to(device), label.to(device)
                out = model(mel, pitch)
                loss = criterion(out, label)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == label).sum().item()
                val_total += label.size(0)

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
                'val_acc': val_acc,
                'epoch': epoch + 1,
            }, best_path)

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Model: {best_path}")

    info = {
        'best_val_acc': best_val_acc,
        'n_train': len(train_items),
        'n_val': len(val_items),
        'class_counts': {string_names[k]: v for k, v in class_counts.items()},
    }
    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def predict_string(mel_spectrogram, midi_pitch, model_path=None):
    """Predict string for a note given its mel spectrogram and MIDI pitch."""
    if model_path is None:
        model_path = os.path.join('..', 'generated', 'string_classifier', 'best_model.pt')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = StringClassifierCNN(NUM_STRINGS)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    x = torch.FloatTensor(mel_spectrogram).unsqueeze(0).unsqueeze(0)
    pitch_norm = (midi_pitch - 30.0) / 60.0
    pitch_tensor = torch.FloatTensor([[pitch_norm]])

    with torch.no_grad():
        out = model(x, pitch_tensor)
        probs = F.softmax(out, dim=1)[0]
        pred_string = probs.argmax().item()
        confidence = probs[pred_string].item()

    return pred_string, confidence, probs.numpy()


def main():
    parser = argparse.ArgumentParser(description='Guitar String Classifier')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
