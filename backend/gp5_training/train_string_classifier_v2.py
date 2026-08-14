"""
Step 3v2: Enhanced String Classifier
=====================================
Improvements over v1:
- Hidden dim: 256 → 512, 5 layers (was 3)
- Sample size: 5M → 10M notes
- Context window: 5 → 8
- NEW features: chord context (simultaneous notes count, pitch range, bass pitch)
- EPOCHS: 30 → 50
Output: gp5_string_classifier.pth (overwrites v1)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
CONTEXT_WINDOW = 8
NUM_STRINGS = 6
BATCH_SIZE = 2048
EPOCHS = 50
LR = 1e-3
HIDDEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StringDataset(Dataset):
    def __init__(self, records):
        self.records = records
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        pitch = r["pitch"]
        pitch_norm = (pitch - 30) / 60.0
        
        # Context pitches
        prev_p = r.get("prev_pitches", [])
        next_p = r.get("next_pitches", [])
        
        def pad(lst, size, default=0):
            lst = lst[:size]
            return lst + [default] * (size - len(lst))
        
        prev_pitches = [(p - 30) / 60.0 for p in pad(prev_p, CONTEXT_WINDOW, 0)]
        next_pitches = [(p - 30) / 60.0 for p in pad(next_p, CONTEXT_WINDOW, 0)]
        
        dur = min(r.get("duration_ticks", 960), 3840) / 3840.0
        vel = min(r.get("velocity", 80), 127) / 127.0
        
        # Possible strings (physical constraint)
        possible = []
        for open_pitch in STANDARD_TUNING:
            f = pitch - open_pitch
            possible.append(1.0 if 0 <= f <= 24 else 0.0)
        
        # NEW: Chord context features
        # How many simultaneous notes, pitch range, bass pitch
        sim_notes = r.get("sim_note_count", 1)
        sim_count_norm = min(sim_notes, 6) / 6.0
        sim_pitches = r.get("sim_pitches", [pitch])
        if sim_pitches:
            pitch_range = (max(sim_pitches) - min(sim_pitches)) / 48.0
            bass_pitch = (min(sim_pitches) - 30) / 60.0
            is_bass = 1.0 if pitch == min(sim_pitches) else 0.0
            is_melody = 1.0 if pitch == max(sim_pitches) else 0.0
        else:
            pitch_range = 0.0
            bass_pitch = pitch_norm
            is_bass = 0.0
            is_melody = 0.0
        
        # Measure position
        measure = r.get("measure", 0)
        measure_norm = (measure % 16) / 16.0
        
        features = (
            [pitch_norm, dur, vel, sim_count_norm, pitch_range, 
             bass_pitch, is_bass, is_melody, measure_norm] +
            prev_pitches + next_pitches +
            possible
        )
        
        target = r["string"] - 1
        if target < 0 or target >= NUM_STRINGS:
            target = 0
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


class StringClassifierNet(nn.Module):
    """Enhanced 5-layer DNN with residual connections"""
    
    def __init__(self, input_dim, hidden=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(4):  # 4 residual blocks
            self.layers.append(nn.Linear(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, NUM_STRINGS),
        )
    
    def forward(self, x):
        x = F.gelu(self.bn0(self.input_proj(x)))
        x = self.dropout(x)
        for layer, bn in zip(self.layers, self.bns):
            residual = x
            x = F.gelu(bn(layer(x)))
            x = self.dropout(x)
            x = x + residual  # Residual connection
        return self.head(x)


def load_data(max_records=10_000_000):
    """Load with reservoir sampling, adding chord context"""
    print(f"  Loading notes (max {max_records:,})...")
    
    # First pass: group simultaneous notes by (measure, time_sec)
    # We'll do this on-the-fly using a sliding approach
    records = []
    total = 0
    kept = 0
    
    # Buffer for simultaneous note detection
    buffer = []
    SIMUL_THRESHOLD = 0.03  # 30ms
    
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
            
            # Add simultaneous note info from buffer
            t = note.get("time_sec", 0)
            # Flush old buffer entries
            buffer = [b for b in buffer if abs(b.get("time_sec", 0) - t) < SIMUL_THRESHOLD]
            
            sim_pitches = [b["pitch"] for b in buffer] + [note["pitch"]]
            note["sim_note_count"] = len(sim_pitches)
            note["sim_pitches"] = sim_pitches
            
            buffer.append(note)
            
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
    print("  Step 3v2: Enhanced String Classifier")
    print(f"  Device: {DEVICE}, Hidden: {HIDDEN}, Context: {CONTEXT_WINDOW}")
    print("=" * 60)
    
    if not NOTES_FILE.exists():
        print(f"  ERROR: {NOTES_FILE} not found.")
        return
    
    print("\n[1/4] Loading dataset...")
    records = load_data()
    
    if len(records) < 1000:
        print(f"  ERROR: Too few records ({len(records)}).")
        return
    
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
    
    sample_x, sample_y = train_ds[0]
    input_dim = sample_x.shape[0]
    print(f"  Train: {len(train_records):,}, Val: {len(val_records):,}")
    print(f"  Input dim: {input_dim}, Output: {NUM_STRINGS}")
    
    print("\n[3/4] Training...")
    model = StringClassifierNet(input_dim, HIDDEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Class weights
    string_counts = [0] * NUM_STRINGS
    for r in train_records:
        string_counts[r["string"] - 1] += 1
    total = sum(string_counts)
    weights = torch.tensor([total / (c + 1) for c in string_counts], dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum() * NUM_STRINGS
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    print(f"  String distribution: {string_counts}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_acc = 0
    patience = 10
    no_improve = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)
        
        scheduler.step()
        
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
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": HIDDEN,
                "num_strings": NUM_STRINGS,
                "context_window": CONTEXT_WINDOW,
                "val_acc": val_acc,
                "epoch": epoch,
                "tuning": STANDARD_TUNING,
                "version": "v2",
            }, MODEL_FILE)
            marker = " ★"
        else:
            no_improve += 1
            marker = ""
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: "
              f"loss={avg_loss:.4f}, train={train_acc:.1f}%, "
              f"val={val_acc:.1f}%{marker} ({elapsed:.0f}s)")
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n[4/4] Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved: {MODEL_FILE}")
    
    # Per-string accuracy
    ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
    model2 = StringClassifierNet(input_dim, HIDDEN).to(DEVICE)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()
    per_string_correct = [0] * NUM_STRINGS
    per_string_total = [0] * NUM_STRINGS
    
    with torch.no_grad():
        for batch_x, batch_y in val_dl:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            preds = model2(batch_x).argmax(dim=1)
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
