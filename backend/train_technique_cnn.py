"""
train_technique_cnn.py - IDMT-SMT-GUITAR_V2 テクニック分類器 (V2: 精度重視)
================================================================================
改良点:
  - 3チャネル入力 (Mel + Δ + ΔΔ) で時間変化を捕捉
  - Focal Loss (class imbalance対策、γ=2)
  - ResNet風アーキテクチャ (Residual blocks + SE attention)
  - Mixup / SpecAugment による拡張
  - 0.5秒コンテキスト窓 (onset前後)
  - 信頼度閾値付きprediction
  - 100 epochs + cosine annealing + early stopping

6クラス: normal, muted, bend, slide, harmonic, vibrato
"""

from __future__ import annotations
import os, sys, json, glob, time, argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─── Config ───
DATASET_ROOT = Path(r"D:\Music\datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2\dataset2")
MODEL_SAVE_PATH = Path(__file__).parent / "models" / "technique_cnn.pth"

SR = 22050
N_MELS = 80           # 64->80 (higher resolution)
N_FFT = 1024
HOP_LENGTH = 256
PATCH_DURATION = 0.5   # 0.3->0.5秒 (longer context)
PATCH_FRAMES = int(PATCH_DURATION * SR / HOP_LENGTH) + 1  # ~44 frames

LABEL_MAP = {
    "normal":    0,
    "muted":     1,
    "bend":      2,
    "slide":     3,
    "harmonic":  4,
    "vibrato":   5,
}
LABEL_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = len(LABEL_MAP)


def _xml_to_label(exc: str, exp: str) -> str:
    if exp == "BE": return "bend"
    elif exp == "SL": return "slide"
    elif exp == "HA": return "harmonic"
    elif exp == "VI": return "vibrato"
    elif exp == "DN": return "muted"
    if exc == "MU": return "muted"
    return "normal"


# ─── Focal Loss ───

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017): down-weight easy examples."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# ─── Dataset ───

class TechniqueDataset(Dataset):
    """3チャネル (Mel + Δ + ΔΔ) パッチを生成。"""

    def __init__(self, entries: list, augment: bool = False, mixup: bool = False):
        self.entries = entries
        self.augment = augment
        self.mixup = mixup
        self._cache = {}
        # Pregroup by label for mixup
        self._by_label = {}
        for i, e in enumerate(entries):
            lbl = LABEL_MAP[e["label"]]
            self._by_label.setdefault(lbl, []).append(i)

    def _load_features(self, wav_path: str):
        """Mel + delta + delta-delta (3チャネル)。"""
        if wav_path not in self._cache:
            import librosa
            y, _ = librosa.load(wav_path, sr=SR, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=50, fmax=8000,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            delta = librosa.feature.delta(mel_db, order=1).astype(np.float32)
            delta2 = librosa.feature.delta(mel_db, order=2).astype(np.float32)
            self._cache[wav_path] = (mel_db, delta, delta2)
        return self._cache[wav_path]

    def _extract_patch(self, wav_path, onset, channel_idx=None):
        mel_db, delta, delta2 = self._load_features(wav_path)
        onset_frame = int(onset * SR / HOP_LENGTH)
        half = PATCH_FRAMES // 2
        start = max(0, onset_frame - half)
        end = start + PATCH_FRAMES
        total_frames = mel_db.shape[1]
        if end > total_frames:
            end = total_frames
            start = max(0, end - PATCH_FRAMES)

        patches = []
        for feat in [mel_db, delta, delta2]:
            p = feat[:, start:end]
            if p.shape[1] < PATCH_FRAMES:
                pad = np.zeros((N_MELS, PATCH_FRAMES - p.shape[1]), dtype=np.float32)
                p = np.concatenate([p, pad], axis=1)
            patches.append(p)

        # Stack: [3, N_MELS, PATCH_FRAMES]
        patch = np.stack(patches, axis=0)
        return patch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        patch = self._extract_patch(e["wav_path"], e["onset"])
        label = LABEL_MAP[e["label"]]

        # Per-channel normalization
        for c in range(3):
            m, s = patch[c].mean(), patch[c].std() + 1e-6
            patch[c] = (patch[c] - m) / s

        # Augmentation
        if self.augment:
            patch = self._augment(patch)

        # Mixup (intra-class only for technique, inter-class for normal)
        if self.mixup and np.random.random() < 0.3:
            patch, label = self._do_mixup(patch, label, idx)

        tensor = torch.from_numpy(patch.copy())
        if isinstance(label, int):
            return tensor, label
        else:
            # Mixup returns soft label
            return tensor, label

    def _augment(self, patch):
        # Time shift ±3 frames
        shift = np.random.randint(-3, 4)
        if shift != 0:
            patch = np.roll(patch, shift, axis=2)

        # Frequency masking (SpecAugment)
        if np.random.random() < 0.5:
            f_width = np.random.randint(4, 12)
            f_start = np.random.randint(0, max(1, N_MELS - f_width))
            patch[:, f_start:f_start+f_width, :] = 0

        # Time masking (SpecAugment)
        if np.random.random() < 0.5:
            t_width = np.random.randint(2, 6)
            t_start = np.random.randint(0, max(1, PATCH_FRAMES - t_width))
            patch[:, :, t_start:t_start+t_width] = 0

        # Gaussian noise
        if np.random.random() < 0.3:
            patch += np.random.randn(*patch.shape).astype(np.float32) * 0.08

        # Random gain ±3dB
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.7, 1.4)
            patch *= gain

        return patch

    def _do_mixup(self, patch, label, idx):
        """Intra-class mixup to avoid label confusion."""
        same_class_indices = self._by_label.get(label, [])
        if len(same_class_indices) < 2:
            return patch, label
        other_idx = idx
        while other_idx == idx:
            other_idx = same_class_indices[np.random.randint(len(same_class_indices))]
        other_e = self.entries[other_idx]
        other_patch = self._extract_patch(other_e["wav_path"], other_e["onset"])
        for c in range(3):
            m, s = other_patch[c].mean(), other_patch[c].std() + 1e-6
            other_patch[c] = (other_patch[c] - m) / s
        lam = np.random.beta(0.4, 0.4)
        mixed = lam * patch + (1 - lam) * other_patch
        return mixed, label  # same class -> hard label


# ─── Model: ResNet-style CNN with SE Attention ───

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.view(b, c, -1).mean(dim=2)  # GAP
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResBlock(nn.Module):
    """Residual block with SE attention."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class TechniqueCNN(nn.Module):
    """ResNet-SE CNN: 3ch input -> 6 class."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─── Data Loading ───

def parse_dataset(dataset_root: Path) -> list:
    annotation_dir = dataset_root / "annotation"
    audio_dir = dataset_root / "audio"
    entries = []
    for xml_path in sorted(glob.glob(str(annotation_dir / "*.xml"))):
        tree = ET.parse(xml_path)
        audio_fname = Path(xml_path).stem + ".wav"
        wav_path = audio_dir / audio_fname
        if not wav_path.exists():
            continue
        for evt in tree.findall(".//event"):
            onset = float(evt.findtext("onsetSec", "0"))
            offset = float(evt.findtext("offsetSec", "0"))
            exc = evt.findtext("excitationStyle", "PK")
            exp = evt.findtext("expressionStyle", "NO")
            label = _xml_to_label(exc, exp)
            entries.append({
                "wav_path": str(wav_path),
                "onset": onset,
                "offset": offset,
                "label": label,
            })
    return entries


def parse_all_datasets(base: Path) -> list:
    entries = []
    ds2 = base.parent / "dataset2"
    if ds2.exists():
        entries.extend(parse_dataset(ds2))
        print(f"  dataset2: {len(entries)} entries")
    ds3 = base.parent / "dataset3"
    if ds3.exists():
        n_before = len(entries)
        entries.extend(parse_dataset(ds3))
        print(f"  dataset3: {len(entries) - n_before} entries")
    print("  dataset4: skipped (CSV chord annotations)")
    return entries


# ─── Training ───

def train(args):
    print("=" * 60)
    print("Technique CNN V2 - Precision-focused training")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] データセット読み込み...")
    entries = parse_all_datasets(DATASET_ROOT)
    print(f"合計: {len(entries)} ノート")

    label_counts = Counter(e["label"] for e in entries)
    print("\nラベル分布:")
    for label in LABEL_NAMES:
        cnt = label_counts.get(label, 0)
        print(f"  {label:12s}: {cnt:5d} ({100*cnt/len(entries):.1f}%)")

    # 2. Split (WAV-level to prevent leakage)
    print("\n[2/5] Train/Val分割...")
    wav_files = list(set(e["wav_path"] for e in entries))
    train_wavs, val_wavs = train_test_split(wav_files, test_size=0.2, random_state=42)
    train_wavs_set = set(train_wavs)
    train_entries = [e for e in entries if e["wav_path"] in train_wavs_set]
    val_entries = [e for e in entries if e["wav_path"] not in train_wavs_set]

    # Oversample minority classes in train set (repeat entries)
    min_class_target = 300  # aim for at least 300 per class
    augmented_train = list(train_entries)
    train_label_counts = Counter(e["label"] for e in train_entries)
    for label in LABEL_NAMES:
        current = train_label_counts.get(label, 0)
        if current < min_class_target and current > 0:
            class_entries = [e for e in train_entries if e["label"] == label]
            repeat_factor = (min_class_target // current) + 1
            augmented_train.extend(class_entries * repeat_factor)
    train_entries = augmented_train
    print(f"Train: {len(train_entries)} (after oversample), Val: {len(val_entries)}")
    train_label_counts2 = Counter(e["label"] for e in train_entries)
    for label in LABEL_NAMES:
        print(f"  Train {label:12s}: {train_label_counts2.get(label, 0)}")

    # 3. DataLoaders
    train_labels = [LABEL_MAP[e["label"]] for e in train_entries]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_entries), replacement=True)

    train_ds = TechniqueDataset(train_entries, augment=True, mixup=True)
    val_ds = TechniqueDataset(val_entries, augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Model
    print("\n[3/5] モデル構築...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = TechniqueCNN(NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Focal Loss with class weights
    total_train = len(train_entries)
    alpha = torch.tensor([
        total_train / (NUM_CLASSES * class_counts.get(i, 1))
        for i in range(NUM_CLASSES)
    ], dtype=torch.float32).to(device)
    # Normalize alpha
    alpha = alpha / alpha.sum() * NUM_CLASSES
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # 5. Training
    epochs = args.epochs
    print(f"\n[4/5] 学習開始 ({epochs} epochs, Focal Loss γ=2)...")
    best_val_f1 = 0.0
    best_epoch = 0
    patience = 25
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            train_correct += (out.argmax(1) == batch_y).sum().item()
            train_total += batch_x.size(0)
        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        train_acc = 100 * train_correct / max(train_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)

        # Macro F1 as primary metric (better for imbalanced data)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        improved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": LABEL_MAP,
                "label_names": LABEL_NAMES,
                "n_mels": N_MELS,
                "patch_frames": PATCH_FRAMES,
                "sr": SR,
                "hop_length": HOP_LENGTH,
                "n_fft": N_FFT,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "num_channels": 3,
            }, str(MODEL_SAVE_PATH))
            improved = f" ★ BEST (F1={val_f1:.3f})"
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Train: {train_acc:.1f}%  "
                  f"Val: {val_acc:.1f}%  "
                  f"F1={val_f1:.3f}{improved}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # 6. Final eval
    print(f"\n[5/5] 最終評価 (Best epoch={best_epoch}, F1={best_val_f1:.3f})...")
    ckpt = torch.load(str(MODEL_SAVE_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            probs = F.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=3))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    header = "          " + " ".join(f"{n[:6]:>6s}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:10s} " + " ".join(f"{v:6d}" for v in row))

    # Confidence analysis
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    print("\n=== 信頼度別精度 ===")
    for thresh in [0.5, 0.7, 0.8, 0.9, 0.95]:
        max_probs = all_probs.max(axis=1)
        mask = max_probs >= thresh
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
            coverage = mask.mean()
            print(f"  閾値>={thresh:.2f}: accuracy={acc:.3f}, coverage={coverage:.1%} ({mask.sum()}/{len(mask)})")

    print(f"\nModel: {MODEL_SAVE_PATH}")
    print(f"Size: {MODEL_SAVE_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args)
