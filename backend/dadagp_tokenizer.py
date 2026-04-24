"""
dadagp_tokenizer.py — DadaGPトークン列→弦予測学習データ変換
============================================================
DadaGPのGuitarProトークン形式から弦予測MLM学習用の
(pitch, time_shift, duration, string) タプル列を抽出する。

DadaGPトークン形式:
  - instrument:note:s{string}:f{fret}  (例: distorted0:note:s4:f2)
  - wait:{ticks}                        (例: wait:480)
  - new_measure
  - nfx:...                             (ノートエフェクト)
  - downtune:{n}                        (半音ダウンチューニング)

弦予測用にピッチ情報のみ使い、弦を予測ターゲットとする。
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Standard tuning: string 1(high E) = 64, string 6(low E) = 40
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

# Guitar instrument prefixes in DadaGP
GUITAR_PREFIXES = ('clean', 'distorted')


@dataclass
class GuitarNote:
    """A single guitar note extracted from DadaGP tokens."""
    pitch: int          # MIDI note number
    string: int         # String number (0-5, 0=high E, 5=low E)
    fret: int           # Fret number
    time_pos: int       # Cumulative time position in ticks
    duration: int       # Duration in ticks (estimated from next event)


@dataclass
class TrainingSequence:
    """A sequence of notes for MLM training."""
    pitches: List[int]        # MIDI pitches
    strings: List[int]        # String indices (0-5), target for MLM
    time_shifts: List[int]    # Time shift from previous note (in ticks)
    durations: List[int]      # Estimated durations (in ticks)
    downtune: int = 0         # Downtune amount in semitones


def parse_dadagp_tokens(filepath: str) -> List[TrainingSequence]:
    """Parse a DadaGP token file and extract guitar note sequences.
    
    Args:
        filepath: Path to .tokens.txt file
        
    Returns:
        List of TrainingSequence, one per guitar track
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Extract metadata
    downtune = 0
    for line in lines:
        if line.startswith('downtune:'):
            try:
                downtune = int(line.split(':')[1])
            except (ValueError, IndexError):
                pass
            break
    
    # Extract notes per guitar track
    track_notes: Dict[str, List[GuitarNote]] = {}
    current_time = 0
    
    for line in lines:
        # Time advance
        if line.startswith('wait:'):
            try:
                ticks = int(line.split(':')[1])
                current_time += ticks
            except (ValueError, IndexError):
                pass
            continue
        
        # Measure boundary (also resets for context)
        if line == 'new_measure':
            continue
        
        # Note parsing: instrument:note:s{string}:f{fret}
        match = re.match(
            r'((?:clean|distorted)\d*):note:s(\d+):f(-?\d+)', line
        )
        if match:
            track_name = match.group(1)
            string_num = int(match.group(2))  # 1-6
            fret = int(match.group(3))
            
            # Calculate MIDI pitch
            if string_num in STANDARD_TUNING:
                pitch = STANDARD_TUNING[string_num] + downtune + fret
            else:
                continue
            
            # Convert string to 0-indexed (0=high E, 5=low E)
            string_idx = string_num - 1
            
            if track_name not in track_notes:
                track_notes[track_name] = []
            
            track_notes[track_name].append(GuitarNote(
                pitch=pitch,
                string=string_idx,
                fret=fret,
                time_pos=current_time,
                duration=0,  # Will be estimated later
            ))
    
    # Convert to TrainingSequences
    sequences = []
    for track_name, notes in track_notes.items():
        if len(notes) < 4:
            continue
        
        # Sort by time position
        notes.sort(key=lambda n: n.time_pos)
        
        # Estimate durations and compute time shifts
        for i in range(len(notes) - 1):
            notes[i].duration = max(1, notes[i + 1].time_pos - notes[i].time_pos)
        notes[-1].duration = 480  # Default last note duration
        
        # Build sequence
        pitches = []
        strings = []
        time_shifts = []
        durations = []
        
        prev_time = 0
        for note in notes:
            pitches.append(note.pitch)
            strings.append(note.string)
            time_shifts.append(note.time_pos - prev_time)
            durations.append(note.duration)
            prev_time = note.time_pos
        
        sequences.append(TrainingSequence(
            pitches=pitches,
            strings=strings,
            time_shifts=time_shifts,
            durations=durations,
            downtune=downtune,
        ))
    
    return sequences


def quantize_time_shift(ticks: int, resolution: int = 480) -> int:
    """Quantize time shift to a discrete bin.
    
    Maps ticks to one of ~32 bins representing common rhythmic values.
    480 ticks = quarter note (standard MIDI resolution).
    
    Returns:
        Quantized bin index (0-31)
    """
    # Common rhythmic fractions of a quarter note
    bins = [
        0, 30, 60, 80, 120, 160, 180, 240,
        320, 360, 480, 640, 720, 960, 1200,
        1440, 1920, 2400, 2880, 3360, 3840,
        4320, 4800, 5760, 7680, 9600, 11520,
        15360, 19200, 23040, 30720, 61440,
    ]
    # Find closest bin
    best_idx = 0
    best_dist = abs(ticks - bins[0])
    for i, b in enumerate(bins):
        dist = abs(ticks - b)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def quantize_duration(ticks: int) -> int:
    """Quantize duration to a discrete bin (same scheme as time_shift)."""
    return quantize_time_shift(ticks)


class DadaGPDataset:
    """PyTorch-compatible dataset for MLM string prediction training.
    
    Loads DadaGP token files and produces training samples for the
    Transformer MLM string predictor.
    """
    
    def __init__(self, token_dir: str, max_seq_len: int = 512,
                 max_files: Optional[int] = None):
        """
        Args:
            token_dir: Directory containing .tokens.txt files (can be nested)
            max_seq_len: Maximum sequence length for training
            max_files: Limit number of files (for debugging)
        """
        self.max_seq_len = max_seq_len
        self.sequences: List[TrainingSequence] = []
        
        # Find all token files
        token_files = []
        for root, dirs, files in os.walk(token_dir):
            for f in files:
                if f.endswith('.tokens.txt'):
                    token_files.append(os.path.join(root, f))
        
        if max_files:
            token_files = token_files[:max_files]
        
        print(f"Loading {len(token_files)} DadaGP token files...")
        
        loaded = 0
        for filepath in token_files:
            try:
                seqs = parse_dadagp_tokens(filepath)
                self.sequences.extend(seqs)
                loaded += 1
            except Exception:
                pass
        
        print(f"Loaded {len(self.sequences)} guitar sequences from {loaded} files")
        
        # Split long sequences into chunks
        self._chunks: List[Tuple[List[int], List[int], List[int], List[int]]] = []
        for seq in self.sequences:
            n = len(seq.pitches)
            for start in range(0, n, max_seq_len):
                end = min(start + max_seq_len, n)
                if end - start < 4:
                    continue
                self._chunks.append((
                    seq.pitches[start:end],
                    seq.strings[start:end],
                    [quantize_time_shift(t) for t in seq.time_shifts[start:end]],
                    [quantize_duration(d) for d in seq.durations[start:end]],
                ))
        
        print(f"Total training chunks: {len(self._chunks)}")
    
    def __len__(self):
        return len(self._chunks)
    
    def __getitem__(self, idx):
        """Returns a dict suitable for the StringPredictor model.
        
        Returns:
            dict with keys:
                'pitches': (seq_len,) int tensor — MIDI pitches
                'time_shifts': (seq_len,) int tensor — quantized time shifts
                'durations': (seq_len,) int tensor — quantized durations
                'strings': (seq_len,) int tensor — target string labels (0-5)
                'length': int — actual sequence length
        """
        import torch
        
        pitches, strings, time_shifts, durations = self._chunks[idx]
        seq_len = len(pitches)
        
        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        
        return {
            'pitches': torch.tensor(pitches + [0] * pad_len, dtype=torch.long),
            'time_shifts': torch.tensor(time_shifts + [0] * pad_len, dtype=torch.long),
            'durations': torch.tensor(durations + [0] * pad_len, dtype=torch.long),
            'strings': torch.tensor(strings + [-1] * pad_len, dtype=torch.long),
            'length': seq_len,
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    import torch
    
    return {
        'pitches': torch.stack([b['pitches'] for b in batch]),
        'time_shifts': torch.stack([b['time_shifts'] for b in batch]),
        'durations': torch.stack([b['durations'] for b in batch]),
        'strings': torch.stack([b['strings'] for b in batch]),
        'lengths': torch.tensor([b['length'] for b in batch], dtype=torch.long),
    }


if __name__ == '__main__':
    # Quick test with example file
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'dadaGP',
            'examples', 'progmetal.gp3.tokens.txt'
        )
    
    print(f"Parsing: {test_file}")
    sequences = parse_dadagp_tokens(test_file)
    
    for i, seq in enumerate(sequences):
        print(f"\nTrack {i}: {len(seq.pitches)} notes")
        print(f"  Pitch range: {min(seq.pitches)}-{max(seq.pitches)}")
        print(f"  String distribution: {dict(zip(*np.unique(seq.strings, return_counts=True)))}")
        print(f"  First 10 notes:")
        for j in range(min(10, len(seq.pitches))):
            print(f"    pitch={seq.pitches[j]:3d} string={seq.strings[j]} "
                  f"fret={seq.pitches[j] - STANDARD_TUNING[seq.strings[j]+1]:2d} "
                  f"time_shift={seq.time_shifts[j]:5d} dur={seq.durations[j]:5d}")
