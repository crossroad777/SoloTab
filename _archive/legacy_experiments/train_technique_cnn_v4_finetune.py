"""
train_technique_cnn_v4_finetune.py - Stage 2: aGPTset fine-tuning
================================================================
Stage 1 (IDMT-only) で学習したモデルを、aGPTsetで低LR微調整。
aGPTsetはアコースティックギターのテクニックデータセット。
テクニックマッピング:
  4 = Natural Harmonics → harmonic
  5 = Palm Mute → muted
  pitched notes without technique → normal
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys, os, csv, glob, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Import from main training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train_technique_cnn import (
    TechniqueCNN, TechniqueDataset, parse_dataset,
    LABEL_MAP, LABEL_NAMES, NUM_CLASSES, SR, N_MELS, N_FFT, HOP_LENGTH,
    PATCH_DURATION, PATCH_FRAMES
)

AGPTSET_ROOT = Path(r"D:\Music\datasets\AG-PT-set\aGPTset")
IDMT_ROOT = Path(r"D:\Music\datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2")
STAGE1_PATH = Path("models/technique_cnn_v4_stage1.pth")
OUTPUT_PATH = Path("models/technique_cnn_v4.pth")

# aGPTset technique ID → our label
AGPTSET_TECHNIQUE_MAP = {
    4: "harmonic",     # Natural Harmonics
    5: "muted",        # Palm Mute
}

def parse_agptset():
    """Parse aGPTset note_labels.csv into training entries."""
    csv_path = AGPTSET_ROOT / "metadata" / "note_labels.csv"
    audio_dir = AGPTSET_ROOT / "audio"
    
    entries = []
    skipped = 0
    
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tech_id = int(row.get('expressive_technique_id', -1))
            onset = float(row.get('onset_label_seconds', 0))
            audio_file = row.get('audio_file_path', '')
            
            # Map technique
            if tech_id in AGPTSET_TECHNIQUE_MAP:
                label = AGPTSET_TECHNIQUE_MAP[tech_id]
            elif tech_id >= 0 and tech_id <= 3:
                # Percussive techniques (kick, snare, tom) → skip
                skipped += 1
                continue
            else:
                # Regular pitched notes → normal
                label = "normal"
            
            # Find audio file
            wav_path = audio_dir / audio_file
            if not wav_path.exists():
                # Try subdirectories
                found = list(audio_dir.rglob(audio_file))
                if found:
                    wav_path = found[0]
                else:
                    skipped += 1
                    continue
            
            entries.append({
                "wav_path": str(wav_path),
                "onset": onset,
                "offset": onset + 0.3,  # Approximate duration
                "label": label,
            })
    
    print(f"  aGPTset: {len(entries)} entries (skipped {skipped} percussive)")
    return entries


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def main():
    print("=" * 60)
    print("V4 Stage 2: Fine-tune on IDMT + aGPTset (low LR)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Stage 1 model
    if not STAGE1_PATH.exists():
        print(f"ERROR: Stage 1 model not found: {STAGE1_PATH}")
        sys.exit(1)
    
    model = TechniqueCNN(num_classes=NUM_CLASSES)
    ckpt = torch.load(str(STAGE1_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    stage1_f1 = ckpt.get('val_f1', 0)
    print(f"Loaded Stage 1: epoch={ckpt.get('epoch')}, val_f1={stage1_f1:.3f}")
    model.to(device)
    
    # Load data: IDMT (keep original domain) + aGPTset (new domain)
    print("\n[1/3] Loading datasets...")
    
    # IDMT data (original training data)
    idmt_entries = []
    for ds_name in ["dataset2", "dataset3"]:
        ds_path = IDMT_ROOT / ds_name
        if ds_path.exists():
            ds_entries = parse_dataset(ds_path)
            idmt_entries.extend(ds_entries)
            print(f"  IDMT {ds_name}: {len(ds_entries)} entries")
    
    # aGPTset data
    agptset_entries = parse_agptset()
    
    # Count technique labels in aGPTset
    agptset_counts = Counter(e["label"] for e in agptset_entries)
    print(f"\n  aGPTset label distribution:")
    for label, cnt in sorted(agptset_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:12s}: {cnt:5d}")
    
    # Combine: all IDMT + aGPTset (harmonic & muted only, plus some normal)
    # To avoid overwhelming IDMT signal, sample aGPTset normal
    agptset_technique = [e for e in agptset_entries if e["label"] != "normal"]
    agptset_normal = [e for e in agptset_entries if e["label"] == "normal"]
    
    # Keep all technique entries, sample normal to match technique count
    n_tech = len(agptset_technique)
    if len(agptset_normal) > n_tech * 3:
        np.random.seed(42)
        indices = np.random.choice(len(agptset_normal), n_tech * 3, replace=False)
        agptset_normal = [agptset_normal[i] for i in indices]
    
    all_entries = idmt_entries + agptset_technique + agptset_normal
    print(f"\n  Combined: {len(all_entries)} entries "
          f"(IDMT={len(idmt_entries)}, aGPTset_tech={len(agptset_technique)}, "
          f"aGPTset_normal={len(agptset_normal)})")
    
    # Split by WAV file
    print("\n[2/3] Train/Val split...")
    wav_files = list(set(e["wav_path"] for e in all_entries))
    train_wavs, val_wavs = train_test_split(wav_files, test_size=0.2, random_state=42)
    train_wavs_set = set(train_wavs)
    train_entries = [e for e in all_entries if e["wav_path"] in train_wavs_set]
    val_entries = [e for e in all_entries if e["wav_path"] not in train_wavs_set]
    
    # Oversample minority classes
    train_counts = Counter(e["label"] for e in train_entries)
    max_count = max(train_counts.values())
    augmented = list(train_entries)
    for label, count in train_counts.items():
        if count < max_count // 3 and count > 0:
            label_entries = [e for e in train_entries if e["label"] == label]
            repeat = max(1, (max_count // 3) // count)
            augmented.extend(label_entries * repeat)
    train_entries = augmented
    
    print(f"  Train: {len(train_entries)}, Val: {len(val_entries)}")
    
    # Datasets
    train_ds = TechniqueDataset(train_entries, augment=True)
    val_ds = TechniqueDataset(val_entries, augment=False)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    
    # Fine-tune with very low LR
    print("\n[3/3] Fine-tuning (60 epochs, lr=1e-5)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = FocalLoss(gamma=2.0)
    
    best_f1 = 0.0
    patience = 15
    no_improve = 0
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, 61):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(device)
                preds = model(X).argmax(1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())
        val_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'stage1_f1': stage1_f1,
                'label_map': LABEL_MAP,
                'label_names': LABEL_NAMES,
                'config': {'n_mels': N_MELS, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH,
                           'sr': SR, 'patch_frames': PATCH_FRAMES},
                'training_info': '2-stage: IDMT→aGPTset fine-tune',
            }, str(OUTPUT_PATH))
            print(f"  Epoch {epoch:3d}/60  Train: {train_acc:.1f}%  Val: {val_acc:.1f}%  "
                  f"F1={val_f1:.3f} ★ BEST (stage1={stage1_f1:.3f})")
        else:
            no_improve += 1
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:3d}/60  Train: {train_acc:.1f}%  Val: {val_acc:.1f}%  "
                      f"F1={val_f1:.3f}")
        
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    print(f"\n{'='*60}")
    print(f"V4 Fine-tune complete!")
    print(f"  Stage 1 F1: {stage1_f1:.3f}")
    print(f"  Stage 2 F1: {best_f1:.3f}")
    print(f"  Improvement: {(best_f1 - stage1_f1)*100:+.1f}pp")
    print(f"  Saved: {OUTPUT_PATH}")
    
    # Final classification report
    if OUTPUT_PATH.exists():
        ckpt = torch.load(str(OUTPUT_PATH), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(device)
                preds = model(X).argmax(1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())
        print(f"\nClassification Report (best model):")
        print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=3))


if __name__ == "__main__":
    main()
