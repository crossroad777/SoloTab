"""
generate_synthetic_tab_data.py — 合成ギタータブ学習データ生成
==============================================================
DadaGPデータセットがない場合の代替として、ギター指板の物理制約を
再現した合成データを大量生成し、MLM弦予測モデルを学習する。

生成アルゴリズム:
1. ランダムなポジション（フレット位置）を選択
2. そのポジション付近で演奏可能なフレーズを生成
3. 和音パターンやスケールパターンも含む
4. 複数のチューニングに対応

これにより、「人間が実際に弾くパターン」を模倣した
弦割り当てデータを大量に生成できる。
"""

import os
import sys
import random
import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass

# Standard tuning: string index 0=high E (64), 5=low E (40)
STANDARD_OPEN = [64, 59, 55, 50, 45, 40]
MAX_FRET = 19

# Common alternate tunings
TUNINGS = {
    'standard':    [64, 59, 55, 50, 45, 40],
    'drop_d':      [64, 59, 55, 50, 45, 38],
    'half_down':   [63, 58, 54, 49, 44, 39],
    'open_d':      [62, 57, 54, 50, 45, 38],
    'open_g':      [62, 59, 55, 50, 43, 38],
    'dadgad':      [62, 57, 55, 50, 45, 38],
}

# Guitar scale patterns (relative fret positions within a position)
SCALE_PATTERNS = {
    'major': [
        [(5, 0), (5, 2), (4, 0), (4, 2), (3, 0), (3, 1),
         (2, 0), (2, 2), (1, 0), (1, 2), (0, 0), (0, 2)],
    ],
    'minor': [
        [(5, 0), (5, 2), (5, 3), (4, 0), (4, 2), (4, 3),
         (3, 0), (3, 2), (2, 0), (2, 1), (2, 3), (1, 0)],
    ],
    'pentatonic_minor': [
        [(5, 0), (5, 3), (4, 0), (4, 2), (3, 0), (3, 2),
         (2, 0), (2, 3), (1, 0), (1, 3), (0, 0), (0, 3)],
    ],
    'pentatonic_major': [
        [(5, 0), (5, 2), (4, 0), (4, 2), (3, 0), (3, 2),
         (2, 0), (2, 2), (1, 0), (1, 2), (0, 0), (0, 2)],
    ],
}

# Common chord voicings (string, relative fret from root)
CHORD_VOICINGS = [
    # Open chords
    [(5, 0), (4, 2), (3, 2), (2, 1), (1, 0), (0, 0)],  # E major
    [(4, 0), (3, 2), (2, 2), (1, 2), (0, 0)],            # A major
    [(3, 0), (2, 2), (1, 3), (0, 2)],                     # D major
    [(5, 0), (4, 2), (3, 0), (2, 0), (1, 0), (0, 0)],    # Em
    [(4, 0), (3, 2), (2, 2), (1, 1), (0, 0)],             # Am
    [(3, 0), (2, 2), (1, 3), (0, 1)],                     # Dm
    # Barre chords (E-shape)
    [(5, 0), (4, 2), (3, 2), (2, 1), (1, 0), (0, 0)],   # Major barre
    [(5, 0), (4, 2), (3, 2), (2, 0), (1, 0), (0, 0)],   # Minor barre
    # Power chords
    [(5, 0), (4, 2), (3, 2)],                              # Power chord
    [(5, 0), (4, 2)],                                      # 5th power chord
]

# Fingerstyle patterns (bass + melody interleaving)
FINGERSTYLE_BASS_STRINGS = [5, 4, 3]  # Low strings for bass
FINGERSTYLE_MELODY_STRINGS = [0, 1, 2]  # High strings for melody


def generate_position_phrase(
    tuning: List[int] = None,
    position: int = None,
    length: int = None,
) -> List[Tuple[int, int]]:
    """Generate a musically plausible phrase at a given position.
    
    Returns list of (string_idx, fret) tuples.
    """
    if tuning is None:
        tuning = STANDARD_OPEN
    if position is None:
        position = random.randint(0, 12)
    if length is None:
        length = random.randint(8, 64)
    
    notes = []
    max_span = 4 if position >= 3 else 3
    
    # Pick notes within the position span
    for _ in range(length):
        string = random.randint(0, 5)
        fret_min = max(0, position - 1)
        fret_max = min(MAX_FRET, position + max_span)
        
        # Include open strings occasionally
        if random.random() < 0.15 and position <= 3:
            fret = 0
        else:
            fret = random.randint(fret_min, fret_max)
        
        notes.append((string, fret))
    
    return notes


def generate_scale_run(
    tuning: List[int] = None,
    position: int = None,
    scale_type: str = None,
    ascending: bool = None,
    length: int = None,
) -> List[Tuple[int, int]]:
    """Generate a scale run (ascending or descending).
    
    Returns list of (string_idx, fret) tuples.
    """
    if tuning is None:
        tuning = STANDARD_OPEN
    if position is None:
        position = random.randint(0, 12)
    if scale_type is None:
        scale_type = random.choice(list(SCALE_PATTERNS.keys()))
    if ascending is None:
        ascending = random.choice([True, False])
    if length is None:
        length = random.randint(8, 24)
    
    pattern = random.choice(SCALE_PATTERNS[scale_type])
    
    # Shift pattern to position
    notes_pool = []
    for string, rel_fret in pattern:
        fret = position + rel_fret
        if 0 <= fret <= MAX_FRET:
            notes_pool.append((string, fret))
    
    if not notes_pool:
        return generate_position_phrase(tuning, position, length)
    
    # Generate run
    if not ascending:
        notes_pool = list(reversed(notes_pool))
    
    notes = []
    idx = 0
    for _ in range(length):
        notes.append(notes_pool[idx % len(notes_pool)])
        idx += 1
        # Occasionally reverse direction
        if random.random() < 0.1:
            notes_pool = list(reversed(notes_pool))
            idx = 0
    
    return notes


def generate_chord_sequence(
    tuning: List[int] = None,
    num_chords: int = None,
) -> List[Tuple[int, int]]:
    """Generate a chord sequence (strummed or arpeggiated).
    
    Returns list of (string_idx, fret) tuples.
    """
    if tuning is None:
        tuning = STANDARD_OPEN
    if num_chords is None:
        num_chords = random.randint(4, 16)
    
    notes = []
    for _ in range(num_chords):
        voicing = random.choice(CHORD_VOICINGS)
        root_fret = random.randint(0, 12)
        
        chord_notes = []
        for string, rel_fret in voicing:
            fret = root_fret + rel_fret
            if 0 <= fret <= MAX_FRET:
                chord_notes.append((string, fret))
        
        if not chord_notes:
            continue
        
        # Arpeggiate or strum
        if random.random() < 0.5:
            # Arpeggiate: add notes one by one
            for cn in chord_notes:
                notes.append(cn)
        else:
            # Strum: add all at once (same time position — will have 0 time shift)
            for cn in chord_notes:
                notes.append(cn)
    
    return notes


def generate_fingerstyle_pattern(
    tuning: List[int] = None,
    length: int = None,
) -> List[Tuple[int, int]]:
    """Generate a fingerstyle pattern with bass + melody separation.
    
    Generates realistic acoustic guitar fingerstyle patterns including:
    - Travis picking (alternating bass)
    - Classical arpeggio (p-i-m-a)
    - Melody-over-bass
    - Open string pedal tones
    """
    if tuning is None:
        tuning = STANDARD_OPEN
    if length is None:
        length = random.randint(16, 64)
    
    position = random.randint(0, 9)
    max_span = 4 if position >= 3 else 3
    
    pattern_type = random.choice([
        'travis', 'arpeggio', 'melody_bass', 'pedal_tone'
    ])
    
    notes = []
    
    if pattern_type == 'travis':
        # Travis picking: alternating bass (5-4 or 6-4) with pinch/melody on high strings
        bass_strings = random.choice([(5, 4), (5, 3), (4, 3)])
        for i in range(length):
            if i % 4 == 0:
                # Beat 1: bass note (root)
                string = bass_strings[0]
                fret = random.choice([0, position]) if position <= 3 else position
            elif i % 4 == 2:
                # Beat 3: alternating bass
                string = bass_strings[1]
                fret = random.choice([0, position, position + 2]) if position <= 3 else position + 2
                fret = min(fret, MAX_FRET)
            else:
                # Beats 2, 4: melody on high strings
                string = random.choice([0, 1, 2])
                fret = random.randint(max(0, position - 1), 
                                      min(MAX_FRET, position + max_span))
            notes.append((string, min(fret, MAX_FRET)))
    
    elif pattern_type == 'arpeggio':
        # Classical arpeggio: p-i-m-a pattern (bass-3rd-2nd-1st strings)
        arpeggio_orders = [
            [5, 2, 1, 0],     # p-i-m-a
            [4, 2, 1, 0],     # p-i-m-a (A bass)
            [5, 2, 1, 0, 1, 2],  # p-i-m-a-m-i
            [5, 2, 0, 1, 0, 2],  # p-i-a-m-a-i
        ]
        order = random.choice(arpeggio_orders)
        for i in range(length):
            string = order[i % len(order)]
            if string >= 3:  # Bass strings
                fret = random.choice([0, position]) if position <= 3 else position
            else:  # Melody strings
                fret = random.randint(max(0, position - 1),
                                      min(MAX_FRET, position + max_span))
            notes.append((string, min(fret, MAX_FRET)))
    
    elif pattern_type == 'melody_bass':
        # Solo guitar: melody on high strings with bass accompaniment
        for i in range(length):
            if i % 3 == 0:
                # Bass
                string = random.choice([5, 4, 3])
                fret = 0 if random.random() < 0.4 else random.randint(
                    max(0, position - 1), min(MAX_FRET, position + 2))
            else:
                # Melody (more stepwise motion)
                string = random.choice([0, 1, 2])
                fret = random.randint(max(0, position), 
                                      min(MAX_FRET, position + max_span))
            notes.append((string, fret))
    
    else:  # pedal_tone
        # Open string pedal tone with melody
        pedal_string = random.choice([5, 4, 0])  # Open bass or high E
        for i in range(length):
            if i % 2 == 0:
                # Pedal tone (always open)
                notes.append((pedal_string, 0))
            else:
                # Moving voice
                string = random.choice([1, 2, 3]) if pedal_string >= 4 else random.choice([3, 4, 5])
                fret = random.randint(max(0, position),
                                      min(MAX_FRET, position + max_span))
                notes.append((string, fret))
    
    return notes


def notes_to_training_sequence(
    notes: List[Tuple[int, int]], 
    tuning: List[int] = None
) -> dict:
    """Convert (string, fret) note list to training sequence dict.
    
    Returns dict with:
        pitches, strings, time_shifts, durations
    """
    if tuning is None:
        tuning = STANDARD_OPEN
    
    pitches = []
    strings = []
    time_shifts = []
    durations = []
    
    for i, (string, fret) in enumerate(notes):
        pitch = tuning[string] + fret
        pitches.append(pitch)
        strings.append(string)
        
        # Simulate realistic time patterns
        if i == 0:
            time_shifts.append(0)
        elif random.random() < 0.15:
            # Simultaneous (chord)
            time_shifts.append(0)
        else:
            # Sequential — common rhythmic values
            ts = random.choice([120, 240, 480, 960, 160, 320, 640])
            time_shifts.append(ts)
        
        dur = random.choice([120, 240, 480, 960, 1920, 160, 320])
        durations.append(dur)
    
    return {
        'pitches': pitches,
        'strings': strings,
        'time_shifts': time_shifts,
        'durations': durations,
    }


class SyntheticTabDataset:
    """PyTorch Dataset that generates synthetic guitar tab data on-the-fly.
    
    Generates large amounts of training data without needing DadaGP.
    Covers:
    - Position-based phrases
    - Scale runs (major, minor, pentatonic)
    - Chord sequences (open, barre, power)
    - Fingerstyle patterns (bass + melody)
    - Multiple tunings
    """
    
    def __init__(self, num_samples: int = 50000, max_seq_len: int = 256,
                 seed: int = 42):
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        
        # Pre-generate sequences for reproducibility
        print(f"Generating {num_samples} synthetic guitar sequences...")
        self.sequences = []
        
        generators = [
            ('position', generate_position_phrase, 0.3),
            ('scale', generate_scale_run, 0.25),
            ('chord', generate_chord_sequence, 0.2),
            ('fingerstyle', generate_fingerstyle_pattern, 0.25),
        ]
        
        for i in range(num_samples):
            # Random tuning
            tuning_name = self.rng.choice(list(TUNINGS.keys()))
            tuning = TUNINGS[tuning_name]
            
            # Random pattern type
            r = self.rng.random()
            cumulative = 0
            for name, gen_fn, prob in generators:
                cumulative += prob
                if r < cumulative:
                    notes = gen_fn(tuning=tuning)
                    break
            else:
                notes = generate_position_phrase(tuning=tuning)
            
            # Limit length
            if len(notes) > max_seq_len:
                notes = notes[:max_seq_len]
            
            seq = notes_to_training_sequence(notes, tuning)
            self.sequences.append(seq)
        
        print(f"Generated {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        from dadagp_tokenizer import quantize_time_shift, quantize_duration
        
        seq = self.sequences[idx]
        seq_len = len(seq['pitches'])
        pad_len = self.max_seq_len - seq_len
        
        return {
            'pitches': torch.tensor(
                seq['pitches'] + [0] * pad_len, dtype=torch.long),
            'time_shifts': torch.tensor(
                [quantize_time_shift(t) for t in seq['time_shifts']] + [0] * pad_len,
                dtype=torch.long),
            'durations': torch.tensor(
                [quantize_duration(d) for d in seq['durations']] + [0] * pad_len,
                dtype=torch.long),
            'strings': torch.tensor(
                seq['strings'] + [-1] * pad_len, dtype=torch.long),
            'length': seq_len,
        }


if __name__ == '__main__':
    # Quick test
    print("=== Synthetic Tab Data Generation Test ===\n")
    
    # Test each generator
    for name, gen_fn in [
        ('Position phrase', generate_position_phrase),
        ('Scale run', generate_scale_run),
        ('Chord sequence', generate_chord_sequence),
        ('Fingerstyle', generate_fingerstyle_pattern),
    ]:
        notes = gen_fn()
        seq = notes_to_training_sequence(notes)
        print(f"{name}: {len(notes)} notes")
        print(f"  Pitch range: {min(seq['pitches'])}-{max(seq['pitches'])}")
        print(f"  Strings used: {sorted(set(seq['strings']))}")
        print(f"  First 5: {[(s, p) for s, p in zip(seq['strings'][:5], seq['pitches'][:5])]}")
        print()
    
    # Test dataset
    dataset = SyntheticTabDataset(num_samples=100, max_seq_len=128)
    item = dataset[0]
    print(f"\nDataset item shapes:")
    print(f"  pitches:     {item['pitches'].shape}")
    print(f"  time_shifts: {item['time_shifts'].shape}")
    print(f"  durations:   {item['durations'].shape}")
    print(f"  strings:     {item['strings'].shape}")
    print(f"  length:      {item['length']}")
    
    # String distribution across dataset
    all_strings = []
    for i in range(len(dataset)):
        s = dataset[i]['strings']
        valid = s[s >= 0]
        all_strings.extend(valid.tolist())
    
    from collections import Counter
    dist = Counter(all_strings)
    print(f"\nString distribution (0=high E, 5=low E):")
    for s in sorted(dist.keys()):
        pct = 100 * dist[s] / len(all_strings)
        print(f"  String {s}: {dist[s]:5d} ({pct:.1f}%)")
