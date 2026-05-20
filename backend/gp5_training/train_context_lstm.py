"""
Step 4: Train Context-Aware Sequence Model (LSTM)
==================================================
Learns hand position and movement patterns from GP5 data.
Input: sequence of (pitch, prev_string, prev_fret) → predict string for each note
Output: gp5_context_lstm.pth
"""
import json, sys, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path(r"D:\Music\nextchord-solotab\backend\gp5_training\data")
MODEL_DIR = Path(r"D:\Music\nextchord-solotab\backend\gp5_training\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NOTES_FILE = DATA_DIR / "notes_dataset.jsonl"
MODEL_FILE = MODEL_DIR / "gp5_context_lstm.pth"

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
NUM_STRINGS = 6
SEQ_LEN = 32  # notes per sequence
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sequences(max_notes=5_000_000):
    """Load notes and group into sequences (by measure/file proximity)"""
    print(f"  Loading notes (first {max_notes:,}) and building sequences...")
    
    # Load standard-tuning notes sequentially (order matters for LSTM!)
    notes = []
    total = 0
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
            notes.append(note)
    
    print(f"  Loaded {len(notes):,} notes (from {total:,} scanned)")
    
    # Build sequences: sliding window of SEQ_LEN notes
    sequences = []
    for i in range(0, len(notes) - SEQ_LEN, SEQ_LEN // 2):
        seq = notes[i:i + SEQ_LEN]
        measures = [n["measure"] for n in seq]
        if max(measures) - min(measures) > SEQ_LEN:
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
        
        # Input features per timestep: pitch, fret, duration, velocity, 
        # possible_strings (6), time_delta
        features = []
        targets = []
        
        prev_time = seq[0].get("time_sec", 0)
        
        for note in seq:
            pitch = (note["pitch"] - 30) / 60.0
            # NOTE: fret is EXCLUDED - it leaks the target string!
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
            
            feat = [pitch, dur, vel, dt] + possible
            features.append(feat)
            targets.append(note["string"] - 1)  # 0-indexed
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        )


class ContextLSTM(nn.Module):
    """Bidirectional LSTM for context-aware string prediction"""
    
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=2, num_strings=6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_strings),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        logits = self.classifier(out)  # (batch, seq_len, num_strings)
        return logits


def main():
    print("=" * 60)
    print("  Step 4: Train Context LSTM")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    
    if not NOTES_FILE.exists():
        print(f"  ERROR: {NOTES_FILE} not found. Run Step 1 first.")
        return
    
    # Load sequences
    print("\n[1/3] Building sequences...")
    sequences = load_sequences()
    
    if len(sequences) < 100:
        print(f"  ERROR: Too few sequences ({len(sequences)}). Need at least 100.")
        return
    
    # Split
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
    
    # Model
    print("\n[3/3] Training LSTM...")
    model = ContextLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, 
                        num_layers=NUM_LAYERS, num_strings=NUM_STRINGS).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    best_val_acc = 0
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
            logits = model(batch_x)  # (batch, seq, classes)
            
            # Flatten for cross entropy
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
        
        # Validate
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
            }, MODEL_FILE)
            marker = " ★"
        else:
            marker = ""
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: "
              f"loss={avg_loss:.4f}, train={train_acc:.1f}%, "
              f"val={val_acc:.1f}%{marker} ({elapsed:.0f}s)")
    
    print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved: {MODEL_FILE}")
    
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {best_val_acc:.1f}% accuracy")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
