"""
Step 4v2: Enhanced Context LSTM
=================================
Improvements over v1:
- SEQ_LEN: 32 → 64
- HIDDEN_DIM: 128 → 256
- NUM_LAYERS: 2 → 3
- max_notes: 5M → 10M (first-N sequential)
- NEW features: chord context (simultaneous note count, pitch range)
- EPOCHS: 20 → 40
Output: gp5_context_lstm.pth (overwrites v1)
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
MODEL_FILE = MODEL_DIR / "gp5_context_lstm.pth"

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
NUM_STRINGS = 6
SEQ_LEN = 64
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
HIDDEN_DIM = 256
NUM_LAYERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sequences(max_notes=10_000_000):
    """Load notes sequentially and build sequences"""
    print(f"  Loading notes (first {max_notes:,}) and building sequences...")
    
    notes = []
    total = 0
    # Buffer for simultaneous note detection
    buffer = []
    SIMUL_THRESHOLD = 0.03
    
    with open(NOTES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            if total % 5_000_000 == 0:
                print(f"    ...scanned {total:,} lines, kept {len(notes):,}")
            if len(notes) >= max_notes:
                break
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
            
            # Add simultaneous note info
            t = note.get("time_sec", 0)
            buffer = [b for b in buffer if abs(b.get("time_sec", 0) - t) < SIMUL_THRESHOLD]
            sim_pitches = [b["pitch"] for b in buffer] + [note["pitch"]]
            note["sim_note_count"] = len(sim_pitches)
            note["sim_pitches"] = sim_pitches
            buffer.append(note)
            
            notes.append(note)
    
    print(f"  Loaded {len(notes):,} notes (from {total:,} scanned)")
    
    # Build sequences with sliding window
    sequences = []
    for i in range(0, len(notes) - SEQ_LEN, SEQ_LEN // 2):
        seq = notes[i:i + SEQ_LEN]
        measures = [n["measure"] for n in seq]
        if max(measures) - min(measures) > SEQ_LEN * 2:
            continue
        sequences.append(seq)
    
    print(f"  Built {len(sequences):,} sequences (len={SEQ_LEN})")
    return sequences


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = []
        targets = []
        
        prev_time = seq[0].get("time_sec", 0)
        
        for note in seq:
            pitch = (note["pitch"] - 30) / 60.0
            dur = min(note.get("duration_ticks", 960), 3840) / 3840.0
            vel = min(note.get("velocity", 80), 127) / 127.0
            
            t = note.get("time_sec", 0)
            dt = min(t - prev_time, 5.0) / 5.0
            prev_time = t
            
            # Physical constraints
            possible = []
            for open_pitch in STANDARD_TUNING:
                f = note["pitch"] - open_pitch
                possible.append(1.0 if 0 <= f <= 24 else 0.0)
            
            # Chord context features
            sim_count = min(note.get("sim_note_count", 1), 6) / 6.0
            sim_pitches = note.get("sim_pitches", [note["pitch"]])
            if sim_pitches:
                pitch_range = (max(sim_pitches) - min(sim_pitches)) / 48.0
                is_bass = 1.0 if note["pitch"] == min(sim_pitches) else 0.0
            else:
                pitch_range = 0.0
                is_bass = 0.0
            
            feat = [pitch, dur, vel, dt, sim_count, pitch_range, is_bass] + possible
            features.append(feat)
            targets.append(note["string"] - 1)
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        )


class ContextLSTM(nn.Module):
    """Enhanced Bidirectional LSTM with attention"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_strings=6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_strings),
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # Self-attention weighting
        attn = self.attention(out)
        out = out * attn
        logits = self.classifier(out)
        return logits


def main():
    print("=" * 60)
    print("  Step 4v2: Enhanced Context LSTM")
    print(f"  Device: {DEVICE}, Hidden: {HIDDEN_DIM}, Layers: {NUM_LAYERS}, SeqLen: {SEQ_LEN}")
    print("=" * 60)
    
    if not NOTES_FILE.exists():
        print(f"  ERROR: {NOTES_FILE} not found.")
        return
    
    print("\n[1/3] Building sequences...")
    sequences = load_sequences()
    
    if len(sequences) < 100:
        print(f"  ERROR: Too few sequences ({len(sequences)}).")
        return
    
    print("\n[2/3] Preparing data...")
    random.shuffle(sequences)
    split = int(len(sequences) * 0.9)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]
    
    train_ds = SequenceDataset(train_seqs)
    val_ds = SequenceDataset(val_seqs)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    
    sample_x, sample_y = train_ds[0]
    input_dim = sample_x.shape[1]
    print(f"  Train: {len(train_seqs):,}, Val: {len(val_seqs):,}")
    print(f"  Input dim: {input_dim}, Seq len: {SEQ_LEN}")
    
    print("\n[3/3] Training LSTM...")
    model = ContextLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                        num_layers=NUM_LAYERS, num_strings=NUM_STRINGS).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
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
            
            B, S, C = logits.shape
            loss = criterion(logits.view(B * S, C), batch_y.view(B * S))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * B * S
            preds = logits.argmax(dim=2)
            train_correct += (preds == batch_y).sum().item()
            train_total += B * S
        
        scheduler.step()
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                preds = logits.argmax(dim=2)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0) * batch_x.size(1)
        
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        avg_loss = train_loss / train_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "num_strings": NUM_STRINGS,
                "seq_len": SEQ_LEN,
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
    
    print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved: {MODEL_FILE}")
    
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {best_val_acc:.1f}% accuracy")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
