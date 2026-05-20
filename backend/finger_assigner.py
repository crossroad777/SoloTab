"""
finger_assigner.py - Hybrid CNN + Constraint Engine v6
=======================================================
Strategy:
  1. CNN predicts finger for each note (83.8% standalone)
  2. Post-processing: enforce biomechanical constraints
     - Open string = finger 0 (override)
     - Chord finger uniqueness
     - Position consistency smoothing (phrase-level)
     - Scale run finger ordering
  3. PDMX table fallback when CNN unavailable
  4. derived_fingering_rules.json for fret-offset → finger mapping
"""
from typing import List, Tuple, Optional
import json
import math
import os
import numpy as np

# PDMX Statistical Table
_PDMX_SF_TABLE = {
    (1,  0): {0: 399, 1: 2, 3: 2, 4: 7},
    (1,  1): {1: 171, 2: 8},
    (1,  2): {2: 97, 1: 155, 3: 77, 4: 36},
    (1,  3): {4: 171, 1: 71, 2: 60, 3: 57},
    (1,  4): {4: 63, 3: 41, 1: 57, 2: 24},
    (1,  5): {4: 153, 1: 101, 3: 18, 2: 54},
    (1,  6): {4: 23, 1: 15, 3: 6, 2: 15},
    (1,  7): {4: 105, 2: 31, 3: 30, 1: 41},
    (1,  8): {4: 40, 1: 20, 3: 4, 2: 23},
    (1,  9): {4: 43, 3: 25, 1: 23, 2: 5},
    (1, 10): {3: 3, 4: 43, 2: 15, 1: 5},
    (1, 11): {4: 10, 3: 14, 2: 3, 1: 4},
    (1, 12): {1: 3, 4: 49, 3: 2, 2: 6},
    (1, 14): {4: 11, 2: 2},
    (1, 15): {4: 5, 2: 3},
    (2,  0): {0: 370, 1: 4, 4: 2, 3: 3, 2: 5},
    (2,  1): {1: 395, 2: 9, 3: 1, 4: 3},
    (2,  2): {1: 162, 2: 132, 4: 12, 3: 47},
    (2,  3): {3: 203, 1: 82, 4: 242, 2: 200},
    (2,  4): {2: 33, 4: 102, 3: 28, 1: 41},
    (2,  5): {4: 55, 1: 77, 3: 26, 2: 52},
    (2,  6): {2: 28, 1: 7, 4: 41, 3: 11},
    (2,  7): {4: 67, 1: 30, 3: 31, 2: 22},
    (2,  8): {4: 56, 2: 15, 1: 3, 3: 7},
    (2,  9): {3: 25, 1: 26, 4: 10, 2: 9},
    (2, 10): {3: 19, 1: 5, 4: 29, 2: 18},
    (2, 11): {2: 4, 4: 11, 1: 4, 3: 4},
    (2, 12): {1: 2, 4: 14, 2: 2, 3: 5},
    (3,  0): {1: 4, 0: 285, 2: 3, 4: 4},
    (3,  1): {2: 8, 1: 147, 3: 2},
    (3,  2): {1: 237, 2: 361, 3: 94, 4: 9},
    (3,  3): {1: 31, 2: 43, 3: 65, 4: 14},
    (3,  4): {1: 45, 2: 47, 3: 70, 4: 30},
    (3,  5): {3: 49, 1: 53, 4: 24, 2: 23},
    (3,  6): {3: 48, 2: 54, 1: 23, 4: 9},
    (3,  7): {3: 75, 2: 10, 4: 11, 1: 22},
    (3,  8): {1: 13, 2: 10, 3: 15, 4: 8},
    (3,  9): {1: 18, 2: 11, 3: 13, 4: 6},
    (3, 10): {4: 3, 3: 10, 1: 2, 2: 4},
    (3, 11): {3: 13, 4: 14, 2: 6},
    (3, 12): {1: 2, 3: 3, 2: 4},
    (4,  0): {0: 238, 4: 3, 2: 3, 1: 1, 3: 2},
    (4,  1): {1: 137, 2: 2},
    (4,  2): {2: 254, 3: 20, 1: 128},
    (4,  3): {2: 26, 3: 185, 1: 40, 4: 19},
    (4,  4): {3: 144, 1: 46, 2: 30, 4: 98},
    (4,  5): {1: 29, 3: 44, 2: 3, 4: 34},
    (4,  6): {2: 18, 3: 31, 1: 26, 4: 19},
    (4,  7): {3: 43, 4: 35, 2: 34, 1: 19},
    (4,  8): {3: 10, 2: 5, 1: 8, 4: 3},
    (4,  9): {4: 10, 3: 15, 2: 5, 1: 9},
    (4, 10): {2: 3, 3: 15, 4: 5, 1: 3},
    (4, 11): {3: 4, 4: 12, 2: 5},
    (4, 12): {3: 5, 4: 1, 1: 2},
    (5,  0): {0: 144, 1: 1},
    (5,  1): {1: 47},
    (5,  2): {1: 133, 2: 163, 3: 7},
    (5,  3): {3: 187, 4: 6, 1: 39, 2: 30},
    (5,  4): {2: 15, 3: 32, 4: 52, 1: 36},
    (5,  5): {2: 10, 4: 40, 3: 42, 1: 12},
    (5,  6): {2: 15, 1: 21, 3: 18, 4: 21},
    (5,  7): {4: 24, 2: 24, 1: 19, 3: 22},
    (5,  8): {2: 3, 1: 2, 3: 19, 4: 1},
    (5,  9): {1: 4, 4: 23, 3: 9, 2: 4},
    (5, 10): {3: 2, 1: 8},
    (5, 11): {2: 2, 4: 1, 3: 5, 1: 2},
    (5, 12): {3: 2, 1: 1},
    (5, 13): {4: 2, 3: 4},
    (6,  0): {0: 34},
    (6,  1): {1: 54},
    (6,  2): {2: 37, 1: 29, 3: 1},
    (6,  3): {1: 20, 3: 95, 2: 50, 4: 8},
    (6,  4): {1: 7, 4: 26, 3: 23, 2: 11},
    (6,  5): {4: 23, 2: 12, 1: 13, 3: 4},
    (6,  6): {2: 4, 4: 17, 1: 13},
    (6,  7): {3: 7, 2: 5, 1: 8, 4: 9},
    (6,  8): {4: 14, 3: 19, 2: 4, 1: 4},
    (6,  9): {4: 17, 1: 2, 2: 1},
}

_PDMX_PROB = {}
for _k, _v in _PDMX_SF_TABLE.items():
    _t = sum(_v.values())
    if _t > 0:
        _PDMX_PROB[_k] = {fg: c / _t for fg, c in _v.items()}

MAX_POS = 19

# ============================================================
# Derived Fingering Rules (from GP5 corpus mining)
# ============================================================
_FRET_OFFSET_RULES = None  # offset → {finger: count}


def _load_derived_rules():
    """Load fret_offset_rules from derived_fingering_rules.json."""
    global _FRET_OFFSET_RULES
    if _FRET_OFFSET_RULES is not None:
        return
    rules_path = os.path.join(os.path.dirname(__file__),
                              'derived_fingering_rules.json')
    if os.path.exists(rules_path):
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            raw = data.get('fret_offset_rules', {})
            _FRET_OFFSET_RULES = {}
            for offset_str, finger_counts in raw.items():
                offset = int(offset_str)
                total = sum(finger_counts.values())
                if total > 0:
                    _FRET_OFFSET_RULES[offset] = {
                        int(fg): cnt / total
                        for fg, cnt in finger_counts.items()
                    }
        except Exception as e:
            print(f"[finger_assigner] derived rules load failed: {e}")
            _FRET_OFFSET_RULES = {}
    else:
        _FRET_OFFSET_RULES = {}


def _finger_from_offset(offset: int) -> Optional[int]:
    """Get most likely finger for a fret offset from position.
    offset = fret - position (0-based).
    Returns finger (1-4) or None."""
    _load_derived_rules()
    assert _FRET_OFFSET_RULES is not None
    probs = _FRET_OFFSET_RULES.get(offset)
    if probs:
        return max(probs, key=probs.get)
    # Standard mapping: offset 0→1, 1→2, 2→3, 3→4
    if 0 <= offset <= 3:
        return offset + 1
    return None

# ============================================================
# CNN Model — Dual-Scale Ensemble (v4 CTX=7 + v5 CTX=15)
# ============================================================
_cnn_models = None  # dict of {ctx: model}
_cnn_device = None
INPUT_DIM = 8
_ENSEMBLE_SCALES = [
    ('finger_cnn_v4.pth', 7, 0.4),
    ('finger_cnn_v5.pth', 15, 0.6),
]


def _load_cnn():
    global _cnn_models, _cnn_device
    if _cnn_models is not None:
        return True
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return False

    class FingerCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(INPUT_DIM, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.conv4 = nn.Conv1d(128, 64, 1)
            self.bn4 = nn.BatchNorm1d(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.4)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, 5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.relu(self.bn1(self.conv1(x)))
            r = x
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            if r.shape == x.shape:
                x = x + r
            x = self.pool(x).squeeze(-1)
            x = self.drop(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    _cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    _cnn_models = {}

    for model_file, ctx, weight in _ENSEMBLE_SCALES:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            # Fallback to best model
            model_path = os.path.join(models_dir, 'finger_cnn_best.pth')
            if not os.path.exists(model_path):
                continue
        m = FingerCNN()
        m.load_state_dict(torch.load(model_path, map_location=_cnn_device,
                                     weights_only=True))
        m.to(_cnn_device)
        m.eval()
        _cnn_models[ctx] = (m, weight)

    return len(_cnn_models) > 0


def _build_features(notes, ctx):
    """Build context-window features for all notes."""
    N = len(notes)
    features = []
    for i in range(N):
        cs = notes[i].get('string') or 3
        cf = notes[i].get('fret') or 0
        if not isinstance(cf, (int, float)): cf = 0
        if not isinstance(cs, (int, float)): cs = 3
        window = []
        for j in range(i - ctx, i + ctx + 1):
            if 0 <= j < N:
                n = notes[j]
                s = n.get('string') or 3
                f = n.get('fret') or 0
                p = n.get('pitch') or 60
                if not isinstance(f, (int, float)): f = 0
                if not isinstance(s, (int, float)): s = 3
                window.append([
                    (s - 3.5) / 3.0, f / 12.0, (p - 60) / 24.0,
                    1.0 if j == i else 0.0, (j - i) / ctx,
                    1.0 if f == 0 else 0.0,
                    (f - cf) / 12.0, (s - cs) / 3.0,
                ])
            else:
                window.append([0] * INPUT_DIM)
        features.append(window)
    return features


def _cnn_predict(notes):
    """Ensemble CNN prediction: returns list of (predicted_finger, prob_array) per note."""
    if not _load_cnn():
        return None
    import torch

    N = len(notes)
    ensemble_probs = np.zeros((N, 5), dtype=np.float32)

    assert _cnn_models is not None
    for ctx, (model, weight) in _cnn_models.items():
        features = _build_features(notes, ctx)
        X = torch.FloatTensor(np.array(features, dtype=np.float32)).to(_cnn_device)
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        ensemble_probs += probs * weight

    preds = ensemble_probs.argmax(axis=1)
    return [(int(preds[i]), ensemble_probs[i]) for i in range(N)]


def _pdmx_predict(string, fret):
    """PDMX fallback: return most likely finger."""
    if fret == 0:
        return 0
    probs = _PDMX_PROB.get((string, fret))
    if probs:
        return max(probs, key=probs.get)
    # Position 1 rule
    if 1 <= fret <= 4:
        return fret
    return 1


def _is_valid_finger(fret, finger, position=None):
    """Check if a finger assignment is biomechanically possible.

    Args:
        fret: Fret number (0 = open string)
        finger: Finger number (0=open, 1=index, 2=middle, 3=ring, 4=pinky)
        position: Optional hand position (fret where index finger sits).
                  If provided, validates that the finger can reach the fret.
    """
    if fret == 0:
        return finger == 0
    if finger == 0:
        return fret == 0
    if not (1 <= finger <= 4):
        return False

    # Position-aware validation
    if position is not None and position >= 1:
        offset = fret - position
        # Finger 1 (index) covers offset 0
        # Finger 2 (middle) covers offset 1
        # Finger 3 (ring) covers offset 2
        # Finger 4 (pinky) covers offset 3
        # Allow ±1 stretch beyond standard position
        expected_offset = finger - 1
        if abs(offset - expected_offset) > 2:
            return False

    return True


def _detect_barre(chord_notes):
    """Detect barre chord: multiple strings on the same fret.
    Returns dict of {fret: [notes]} for frets with 2+ notes."""
    from collections import defaultdict
    fret_groups = defaultdict(list)
    for note in chord_notes:
        fret = note.get('fret', 0)
        if fret > 0:
            fret_groups[fret].append(note)
    return {f: notes for f, notes in fret_groups.items() if len(notes) >= 2}


def _resolve_chord_conflicts(chord_notes):
    """Ensure valid finger assignments in chords.

    Handles:
    1. Barre detection: same fret on 2+ strings → finger 1 (index barre)
    2. Finger uniqueness: no two non-barre notes share a finger
    3. Finger ordering: fret(I) ≤ fret(M) ≤ fret(R) ≤ fret(P)
    """
    if not chord_notes:
        return

    # --- Phase 1: Barre detection ---
    barres = _detect_barre(chord_notes)
    barre_notes = set()

    for barre_fret, barre_group in barres.items():
        # The lowest fret in the chord that has 2+ strings is the barre fret
        # Assign finger 1 (index) to all notes on this fret
        for note in barre_group:
            note['left_hand_finger'] = 1
            barre_notes.add(id(note))

    # --- Phase 2: Assign non-barre notes ---
    # Sort non-barre fretted notes by fret ascending
    non_barre = [n for n in chord_notes
                 if id(n) not in barre_notes and n.get('fret', 0) > 0]
    non_barre.sort(key=lambda n: n.get('fret', 0))

    # Determine position from barre fret or min fret
    if barres:
        position = min(barres.keys())
    else:
        fretted = [n.get('fret', 0) for n in chord_notes if n.get('fret', 0) > 0]
        position = min(fretted) if fretted else 1

    # Available fingers (1 may be used for barre)
    used_fingers = {1} if barres else set()

    for note in non_barre:
        fret = note.get('fret', 0)
        if fret <= 0:
            continue

        offset = fret - position
        # Ideal finger based on offset
        if 0 <= offset <= 3:
            ideal = offset + 1
        else:
            ideal = note.get('left_hand_finger', 2)

        # If ideal finger is available and valid, use it
        if ideal not in used_fingers and 1 <= ideal <= 4:
            note['left_hand_finger'] = ideal
            used_fingers.add(ideal)
        else:
            # Find best available finger
            probs = note.get('_finger_probs')
            assigned = False
            if probs is not None:
                order = np.argsort(-probs)
                for alt in order:
                    alt = int(alt)
                    if alt not in used_fingers and 1 <= alt <= 4:
                        note['left_hand_finger'] = alt
                        used_fingers.add(alt)
                        assigned = True
                        break
            if not assigned:
                for alt in [2, 3, 4, 1]:
                    if alt not in used_fingers:
                        note['left_hand_finger'] = alt
                        used_fingers.add(alt)
                        break

    # --- Phase 3: Enforce finger ordering ---
    # fret(finger_i) <= fret(finger_j) when finger_i < finger_j
    _enforce_chord_finger_order(chord_notes)


def _enforce_chord_finger_order(chord_notes):
    """Enforce anatomical constraint: fret(I) ≤ fret(M) ≤ fret(R) ≤ fret(P).
    If violated, swap finger assignments to satisfy ordering."""
    fretted = [(n, n.get('fret', 0), n.get('left_hand_finger', 0))
               for n in chord_notes if n.get('fret', 0) > 0 and n.get('left_hand_finger', 0) > 0]
    if len(fretted) < 2:
        return

    # Sort by finger number
    fretted.sort(key=lambda x: x[2])

    # Check ordering: each finger's fret should be >= previous finger's fret
    for i in range(1, len(fretted)):
        prev_note, prev_fret, prev_finger = fretted[i - 1]
        curr_note, curr_fret, curr_finger = fretted[i]
        if curr_fret < prev_fret and prev_finger < curr_finger:
            # Violation: swap finger assignments
            prev_note['left_hand_finger'] = curr_finger
            curr_note['left_hand_finger'] = prev_finger
            fretted[i - 1] = (prev_note, prev_fret, curr_finger)
            fretted[i] = (curr_note, curr_fret, prev_finger)


def _estimate_position(fretted_notes: List[dict]) -> Optional[int]:
    """Estimate hand position from a group of fretted notes.
    Position = fret where index finger (finger 1) would be.
    Uses weighted median of high-confidence assignments."""
    frets = []
    weights = []
    for n in fretted_notes:
        fret = n.get('fret', 0)
        if not isinstance(fret, (int, float)) or fret <= 0:
            continue
        finger = n.get('left_hand_finger', 1)
        conf = n.get('_finger_conf', 0.5)
        if finger <= 0:
            continue
        # position = fret - (finger - 1)
        pos = int(fret) - (finger - 1)
        if pos >= 1:
            frets.append(pos)
            weights.append(conf)
    if not frets:
        return None
    # Weighted median
    total = sum(weights)
    if total <= 0:
        return int(np.median(frets))
    # Sort by position, find weighted median
    pairs = sorted(zip(frets, weights))
    cumsum = 0.0
    for pos, w in pairs:
        cumsum += w
        if cumsum >= total / 2:
            return pos
    return pairs[-1][0]


def _position_smoothing(notes: List[dict], phrase_gap: float = 0.5,
                        conf_threshold: float = 0.5) -> int:
    """Step 3: Position consistency smoothing.

    Strategy: "Decide position first, fingers follow."
    1. Split notes into phrases (by gap > phrase_gap)
    2. Within each phrase, use a sliding window to find the dominant position
    3. Apply position-consistent fingering to ALL notes in the window
       (not just low-confidence ones)

    This matches how guitarists actually play: they choose a hand position
    and keep it until a position shift is needed.

    Returns number of notes reassigned."""
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))

    # --- Phase 1: Split into phrases ---
    phrases: List[List[dict]] = []
    current_phrase: List[dict] = []
    for note in sorted_notes:
        if not current_phrase:
            current_phrase.append(note)
            continue
        gap = note.get('start', 0) - current_phrase[-1].get('start', 0)
        if gap > phrase_gap:
            phrases.append(current_phrase)
            current_phrase = [note]
        else:
            current_phrase.append(note)
    if current_phrase:
        phrases.append(current_phrase)

    reassigned = 0

    for phrase in phrases:
        # Only fretted notes participate in position estimation
        fretted = [n for n in phrase
                   if isinstance(n.get('fret', 0), (int, float))
                   and n.get('fret', 0) > 0]
        if len(fretted) < 2:
            continue

        # --- Phase 2: Segment into position-consistent groups ---
        # Find the best position for each sub-segment where all frets
        # can be covered within a 4-fret span
        segments = _segment_by_position(fretted)

        for seg_notes, seg_pos in segments:
            if seg_pos < 1:
                continue
            for note in seg_notes:
                fret = int(note.get('fret', 0))
                if fret <= 0:
                    continue
                offset = fret - seg_pos
                if 0 <= offset <= 3:
                    ideal_finger = _finger_from_offset(offset)
                    if ideal_finger is None:
                        continue
                    if not _is_valid_finger(fret, ideal_finger):
                        continue
                    current = note.get('left_hand_finger', 0)
                    if current != ideal_finger:
                        note['left_hand_finger'] = ideal_finger
                        reassigned += 1

    return reassigned


def _segment_by_position(fretted_notes: List[dict]) -> List[tuple]:
    """Split fretted notes into segments that each fit in one hand position.

    Returns list of (notes_in_segment, position) tuples.

    Algorithm: greedy forward scan. Start with position = min_fret of first note.
    Extend segment while all frets fit in [pos, pos+3]. When a note doesn't fit,
    start a new segment.
    """
    if not fretted_notes:
        return []

    segments = []
    seg_start = 0

    while seg_start < len(fretted_notes):
        # Determine initial position from the first fretted note
        first_fret = int(fretted_notes[seg_start].get('fret', 1))
        # Try different positions and pick the one that covers the most notes
        best_pos = first_fret
        best_end = seg_start

        # Try positions from first_fret-3 to first_fret
        for candidate_pos in range(max(1, first_fret - 3), first_fret + 1):
            end = seg_start
            for k in range(seg_start, len(fretted_notes)):
                fret = int(fretted_notes[k].get('fret', 0))
                offset = fret - candidate_pos
                if 0 <= offset <= 3:
                    end = k
                else:
                    break
            if end > best_end or (end == best_end and candidate_pos <= best_pos):
                best_end = end
                best_pos = candidate_pos

        seg_notes = fretted_notes[seg_start:best_end + 1]
        segments.append((seg_notes, best_pos))
        seg_start = best_end + 1

    return segments


def _smooth_scale_runs(notes: List[dict]) -> int:
    """Step 4: Enforce finger ordering on same-string consecutive runs.

    Handles two patterns:
    a) Monotonic runs: ascending/descending frets → position-based fingers
    b) Oscillating patterns: hammer-on/pull-off (e.g. 5f→7f→5f→7f)
       → consistent position with fixed finger per fret

    Position is computed directly from fret values (not from CNN predictions)
    to avoid circular dependency on possibly-wrong finger assignments.

    Returns number of notes corrected."""
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    corrected = 0

    i = 0
    while i < len(sorted_notes) - 1:
        # Find runs of same-string consecutive fretted notes
        run = [sorted_notes[i]]
        j = i + 1
        while j < len(sorted_notes):
            curr = sorted_notes[j]
            prev = run[-1]
            # Same string, close in time
            if (curr.get('string') == prev.get('string') and
                    curr.get('string') is not None and
                    abs(curr.get('start', 0) - prev.get('start', 0)) < 0.4 and
                    curr.get('fret', 0) > 0 and prev.get('fret', 0) > 0):
                run.append(curr)
                j += 1
            else:
                break

        if len(run) >= 2:
            frets = [int(n.get('fret', 0)) for n in run]
            min_fret = min(frets)
            max_fret = max(frets)
            span = max_fret - min_fret

            # Check patterns
            ascending = all(frets[k] <= frets[k+1] for k in range(len(frets)-1))
            descending = all(frets[k] >= frets[k+1] for k in range(len(frets)-1))
            # Oscillating: uses only 2-3 distinct frets (hammer/pull-off)
            unique_frets = set(frets)
            oscillating = len(unique_frets) <= 3 and len(run) >= 3

            if (ascending or descending or oscillating) and span <= 4:
                # All notes fit in one position: pos = min_fret
                pos = min_fret

                if pos >= 1:
                    for note in run:
                        fret = int(note.get('fret', 0))
                        offset = fret - pos
                        if 0 <= offset <= 3:
                            ideal_finger = offset + 1
                            current = note.get('left_hand_finger', 0)
                            if current != ideal_finger and _is_valid_finger(fret, ideal_finger):
                                note['left_hand_finger'] = ideal_finger
                                corrected += 1

            elif (ascending or descending) and span > 4:
                # Span > 4: apply position from min_fret, fix what fits
                pos = min_fret
                if pos >= 1:
                    for note in run:
                        fret = int(note.get('fret', 0))
                        offset = fret - pos
                        if 0 <= offset <= 3:
                            ideal_finger = offset + 1
                            current = note.get('left_hand_finger', 0)
                            if current != ideal_finger and _is_valid_finger(fret, ideal_finger):
                                note['left_hand_finger'] = ideal_finger
                                corrected += 1

        i = j

    return corrected


def assign_fingers(notes: List[dict], phrase_gap: float = 0.5,
                    techniques: Optional[List[str]] = None) -> List[dict]:
    """Main API: Assign left_hand_finger (0-4) to each note.
    CNN-first with biomechanical post-processing.

    Args:
        notes: List of note dicts with string, fret, pitch, start keys
        phrase_gap: Gap in seconds to split phrases for smoothing
        techniques: Optional list of technique strings (1:1 with notes).
                    Values: 'normal', 'hammer_on', 'pull_off', 'slide_up',
                    'slide_down', 'bend', 'harmonic', etc.
    """
    if not notes:
        return notes

    # Attach technique info to notes if provided
    if techniques and len(techniques) == len(notes):
        for i, note in enumerate(notes):
            note['_technique'] = techniques[i]

    # Step 1: CNN prediction
    cnn_results = _cnn_predict(notes)
    use_cnn = cnn_results is not None

    for i, note in enumerate(notes):
        fret = note.get('fret', 0) or 0
        if not isinstance(fret, (int, float)):
            fret = 0

        if fret == 0:
            note['left_hand_finger'] = 0
            note['_finger_conf'] = 1.0
            continue

        if use_cnn and cnn_results is not None:
            pred, probs = cnn_results[i]
            note['_finger_probs'] = probs

            # Validate CNN prediction
            if _is_valid_finger(fret, pred):
                note['left_hand_finger'] = pred
                note['_finger_conf'] = float(probs[pred])
            else:
                # CNN gave invalid finger: pick best valid one
                order = np.argsort(-probs)
                assigned = False
                for alt in order:
                    alt = int(alt)
                    if _is_valid_finger(fret, alt):
                        note['left_hand_finger'] = alt
                        note['_finger_conf'] = float(probs[alt])
                        assigned = True
                        break
                if not assigned:
                    note['left_hand_finger'] = _pdmx_predict(
                        note.get('string', 3), fret)
                    note['_finger_conf'] = 0.5
        else:
            note['left_hand_finger'] = _pdmx_predict(
                note.get('string', 3), fret)
            note['_finger_conf'] = 0.5

    # Step 2: Group simultaneous notes and resolve chord conflicts
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    groups = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if note.get('start', 0) - groups[-1][0].get('start', 0) <= 0.03:
            groups[-1].append(note)
        else:
            groups.append([note])

    for group in groups:
        if len(group) > 1:
            _resolve_chord_conflicts(group)

    # Step 3: Position consistency smoothing
    smoothed = _position_smoothing(notes, phrase_gap=phrase_gap)

    # Step 4: Scale run finger ordering
    run_fixes = _smooth_scale_runs(notes)

    # Step 5: Technique-aware finger constraints
    tech_fixes = _apply_technique_constraints(notes)

    # Cleanup temp attributes
    for note in notes:
        note.pop('_finger_conf', None)
        note.pop('_finger_probs', None)
        note.pop('_technique', None)

    mode = "CNN" if use_cnn else "PDMX"
    print(f"[finger_assigner] {len(notes)} notes ({mode}, "
          f"{len(groups)} groups, smoothed={smoothed}, run_fixes={run_fixes}, "
          f"tech_fixes={tech_fixes})")
    return notes


def _apply_technique_constraints(notes: List[dict]) -> int:
    """Step 5: Apply technique-specific finger constraints.

    Rules:
    - slide_up/slide_down: Same finger on source and target notes
    - hammer_on: Target finger must be HIGHER than source (higher fret = higher finger)
    - pull_off: Target finger must be LOWER than source (lower fret = lower finger)
    - bend: Prefer ring finger (3) — strongest for bending

    Only modifies notes where the technique constraint conflicts with
    the current assignment. Minimal intervention approach.

    Returns number of notes corrected."""
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    corrected = 0

    for i in range(len(sorted_notes)):
        note = sorted_notes[i]
        tech = note.get('_technique', 'normal')
        fret = note.get('fret', 0)
        finger = note.get('left_hand_finger', 0)

        if tech == 'normal' or fret == 0 or finger == 0:
            continue

        # --- Bend: prefer ring finger (3) ---
        if tech == 'bend':
            if finger != 3 and _is_valid_finger(fret, 3):
                note['left_hand_finger'] = 3
                corrected += 1
            continue

        # --- Slide: same finger on connected notes ---
        if tech in ('slide_up', 'slide_down'):
            # Find the previous fretted note (slide source)
            prev = _find_prev_fretted(sorted_notes, i)
            if prev is not None:
                prev_finger = prev.get('left_hand_finger', 0)
                if prev_finger > 0 and finger != prev_finger:
                    if _is_valid_finger(fret, prev_finger):
                        note['left_hand_finger'] = prev_finger
                        corrected += 1
            continue

        # --- Hammer-on: target finger must be higher ---
        if tech == 'hammer_on':
            prev = _find_prev_fretted(sorted_notes, i)
            if prev is not None:
                prev_finger = prev.get('left_hand_finger', 0)
                prev_fret = prev.get('fret', 0)
                # Hammer-on goes to higher fret → need higher finger
                if prev_finger > 0 and fret > prev_fret and finger <= prev_finger:
                    # Strategy 1: raise target finger
                    raised = False
                    for candidate in range(prev_finger + 1, 5):
                        if _is_valid_finger(fret, candidate):
                            note['left_hand_finger'] = candidate
                            corrected += 1
                            raised = True
                            break
                    # Strategy 2: if can't raise target (prev=pinky), lower source
                    if not raised:
                        offset = fret - prev_fret
                        # Set source to index, target based on offset
                        ideal_src = 1
                        ideal_tgt = min(ideal_src + offset, 4)
                        if ideal_tgt > ideal_src:
                            prev['left_hand_finger'] = ideal_src
                            note['left_hand_finger'] = ideal_tgt
                            corrected += 1
            continue

        # --- Pull-off: target finger must be lower ---
        if tech == 'pull_off':
            prev = _find_prev_fretted(sorted_notes, i)
            if prev is not None:
                prev_finger = prev.get('left_hand_finger', 0)
                prev_fret = prev.get('fret', 0)
                # Pull-off goes to lower fret → need lower finger
                if prev_finger > 0 and fret < prev_fret and finger >= prev_finger:
                    for candidate in range(prev_finger - 1, 0, -1):
                        if _is_valid_finger(fret, candidate):
                            note['left_hand_finger'] = candidate
                            corrected += 1
                            break
            continue

    return corrected


def _find_prev_fretted(sorted_notes: List[dict], current_idx: int) -> Optional[dict]:
    """Find the previous fretted note (skipping open strings)."""
    for j in range(current_idx - 1, -1, -1):
        if sorted_notes[j].get('fret', 0) > 0:
            return sorted_notes[j]
    return None

