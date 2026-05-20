"""
Step 3: Train New String Classifier
=====================================
Train a deep neural network to predict guitar string from pitch + context
Using 17,000+ GP5 parsed data (vs previous 360 songs from GuitarSet)
Output: gp5_string_classifier.pth
"""
import json, sys, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(r"D:\Music\nextchord-solotab\backend\gp5_training\data")
MODEL_DIR = Path(r"D:\Music\nextchord-solotab\backend\gp5_training\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NOTES_FILE = DATA_DIR / "notes_dataset.jsonl"
MODEL_FILE = MODEL_DIR / "gp5_string_classifier.pth"

# Standard tuning only for training
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
CONTEXT_WINDOW = 5
NUM_STRINGS = 6
BATCH_SIZE = 2048
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StringDataset(Dataset):
    """Dataset for string prediction from pitch + context"""
    
    def __init__(self, records):
        self.records = records
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        
        # Features: pitch (normalized), prev/next pitches, possible_strings
        # NOTE: fret and prev/next strings are EXCLUDED - they leak the target!
        pitch = r["pitch"]
        
        # Normalize pitch to 0-1 range (MIDI 30-90)
        pitch_norm = (pitch - 30) / 60.0
        
        # Context pitches (padded to CONTEXT_WINDOW) - only pitches, not strings!
        prev_p = r.get("prev_pitches", [])
        next_p = r.get("next_pitches", [])
        
        # Pad/truncate
        def pad(lst, size, default=0):
            lst = lst[:size]
            return lst + [default] * (size - len(lst))
        
        prev_pitches = [(p - 30) / 60.0 for p in pad(prev_p, CONTEXT_WINDOW, 0)]
        next_pitches = [(p - 30) / 60.0 for p in pad(next_p, CONTEXT_WINDOW, 0)]
        
        # Duration feature
        dur = min(r.get("duration_ticks", 960), 3840) / 3840.0
        
        # Velocity
        vel = min(r.get("velocity", 80), 127) / 127.0
        
        # Possible strings for this pitch (physical constraint)
        possible = []
        for s_idx, open_pitch in enumerate(STANDARD_TUNING):
            f = pitch - open_pitch
            if 0 <= f <= 24:
                possible.append(1.0)
            else:
                possible.append(0.0)
        
        features = (
            [pitch_norm, dur, vel] +
            prev_pitches + next_pitches +
            possible
        )
        
        # Target: string (1-indexed → 0-indexed)
        target = r["string"] - 1
        if target < 0 or target >= NUM_STRINGS:
            target = 0
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


class StringClassifierNet(nn.Module):
    """Deep network for string prediction"""
    
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden // 2, NUM_STRINGS),
        )
    
    def forward(self, x):
        return self.net(x)


def load_data(max_records=5_000_000):
    """Load and filter notes for standard tuning with reservoir sampling"""
    print(f"  Loading notes (max {max_records:,})...")
    records = []
    total = 0
    kept = 0
    with open(NOTES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            if total % 5_000_000 == 0:
                print(f"    ...scanned {total:,} lines, kept {kept:,}")
            try:
                note = json.loads(line)
            except Exception:
                continue
            tuning = note.get("tuning", [])[:6]
            if tuning != STANDARD_TUNING:
                continue
            if note["string"] < 1 or note["string"] > 6:
                continue
            if note["fret"] < 0 or note["fret"] > 24:
                continue
            
            # Reservoir sampling
            kept += 1
            if len(records) < max_records:
                records.append(note)
            else:
                j = random.randint(0, kept - 1)
                if j < max_records:
                    records[j] = note
    
    print(f"  Total notes: {total:,}, standard tuning: {kept:,}, sampled: {len(records):,}")
    return records


def main():
    print("=" * 60)
    print("  Step 3: Train String Classifier")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    
    if not NOTES_FILE.exists():
        print(f"  ERROR: {NOTES_FILE} not found. Run Step 1 first.")
        return
    
    # Load data
    print("\n[1/4] Loading dataset...")
    records = load_data()
    
    if len(records) < 1000:
        print(f"  ERROR: Too few records ({len(records)}). Need at least 1000.")
        return
    
    # Shuffle and split
    print("\n[2/4] Preparing train/val split...")
    random.shuffle(records)
    split = int(len(records) * 0.9)
    train_records = records[:split]
    val_records = records[split:]
    
    train_ds = StringDataset(train_records)
    val_ds = StringDataset(val_records)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                        num_workers=0, pin_memory=True)
    
    # Check input dimension
    sample_x, sample_y = train_ds[0]
    input_dim = sample_x.shape[0]
    print(f"  Train: {len(train_records):,}, Val: {len(val_records):,}")
    print(f"  Input dim: {input_dim}, Output: {NUM_STRINGS}")
    
    # Model
    print("\n[3/4] Training...")
    model = StringClassifierNet(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Class weights (handle imbalance)
    string_counts = [0] * NUM_STRINGS
    for r in train_records:
        string_counts[r["string"] - 1] += 1
    total = sum(string_counts)
    weights = torch.tensor([total / (c + 1) for c in string_counts], dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum() * NUM_STRINGS
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    print(f"  String distribution: {string_counts}")
    print(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
        
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        avg_loss = train_loss / train_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": 256,
                "num_strings": NUM_STRINGS,
                "context_window": CONTEXT_WINDOW,
                "val_acc": val_acc,
                "epoch": epoch,
                "tuning": STANDARD_TUNING,
            }, MODEL_FILE)
            marker = " ★"
        else:
            marker = ""
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: "
              f"loss={avg_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_acc={val_acc:.1f}%{marker} ({elapsed:.0f}s)")
    
    print(f"\n[4/4] Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved: {MODEL_FILE}")
    
    # Per-string accuracy
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True)["model_state_dict"])
    model.eval()
    per_string_correct = [0] * NUM_STRINGS
    per_string_total = [0] * NUM_STRINGS
    
    with torch.no_grad():
        for batch_x, batch_y in val_dl:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            preds = model(batch_x).argmax(dim=1)
            for s in range(NUM_STRINGS):
                mask = batch_y == s
                per_string_total[s] += mask.sum().item()
                per_string_correct[s] += ((preds == batch_y) & mask).sum().item()
    
    print(f"\n  Per-string accuracy:")
    for s in range(NUM_STRINGS):
        acc = per_string_correct[s] / per_string_total[s] * 100 if per_string_total[s] > 0 else 0
        print(f"    String {s+1}: {acc:.1f}% ({per_string_correct[s]}/{per_string_total[s]})")
    
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {best_val_acc:.1f}% accuracy")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
