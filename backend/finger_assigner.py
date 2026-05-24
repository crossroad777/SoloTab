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
# Viterbi DP Weights for Finger Assignment (v8.3)
# Refs: Sayegh 1989, Miura 2003, Hori & Sagayama 2016,
#       Radicioni & Lombardo 2005, Carlevaro Technique
# ============================================================
_FINGER_DP_WEIGHTS = {
    # Optuna Phase 6: co-optimized with ctx=5+7 ensemble (99.0%/99.5%)
    'w_cnn_prior': 34.99,          # CNN trust: 4→21→29→35
    'w_offset_rule': 8.89,
    'w_std_offset': 0.44,
    'w_position_same': -7.46,
    'w_position_shift': 7.81,
    'w_position_shift_free': 1.07,
    'w_finger_cross': 200.0,
    'w_same_finger_diff': 6.71,
    'w_span_excess': 2.95,
    'w_tendon_coupling': 0.77,
    'w_continuity_2fret': -6.92,
    'w_guide_finger': -19.81,
    'w_minimax_threshold': 91.76,
    'w_minimax_excess': 3.20,
    'w_barre_continuity': -0.61,
    'w_anchor_penalty': 7.58,
    'w_chord_pos_bonus': -23.47,
    'w_string_cross': 9.22,
    'w_voice_cross_discount': 0.83,
    'w_slide_shift_bonus': -5.76,
    'w_finger_order': 0.0,
    'w_finger_pair_smooth': 0.0,
}

# Chord weights from Optuna Phase 6
_CHORD_DP_WEIGHTS = {
    'w_cnn_prior': 10.40,          # Chord CNN trust rising (3.79→10.40)
    'w_offset_rule': 23.72,
    'w_std_offset': 3.45,
    'w_position_same': -7.25,
    'w_position_shift': 15.33,
    'w_position_shift_free': 2.41,
    'w_finger_cross': 200.0,
    'w_same_finger_diff': 23.91,
    'w_span_excess': 16.70,
    'w_tendon_coupling': 13.32,
    'w_continuity_2fret': -7.67,
    'w_guide_finger': -24.37,
    'w_minimax_threshold': 87.75,
    'w_minimax_excess': 1.64,
    'w_barre_continuity': -16.99,
    'w_anchor_penalty': 39.54,
    'w_chord_pos_bonus': -17.98,
    'w_string_cross': 1.36,
    'w_voice_cross_discount': 0.39,
    'w_slide_shift_bonus': -12.82,
    'w_finger_order': 0.0,
    'w_finger_pair_smooth': 0.0,
}

# Active weights pointer (set per-phrase by context blending)
_ACTIVE_WEIGHTS = _FINGER_DP_WEIGHTS

# Maximum comfortable span (in frets) between two fingers
_BIO_MAX_SPAN = {
    (1, 2): 4,
    (1, 3): 5,
    (1, 4): 6,
    (2, 3): 3,
    (2, 4): 4,
    (3, 4): 3,
}


def _position_adjusted_max_span(finger_lo: int, finger_hi: int,
                                position: int) -> int:
    """Adjust maximum finger span based on fretboard position.

    Guitar fret spacing follows 12-TET: spacing(n) = L / 2^(n/12).
    At fret 1, spacing ≈ 36mm; at fret 12, spacing ≈ 18mm.
    Lower positions have wider frets → less stretch possible.
    Higher positions have narrower frets → more stretch possible.

    Ref: Miura et al. 2003 — "wrist movement minimization"
         emphasizes position-dependent ergonomics.
    """
    base_span = _BIO_MAX_SPAN.get((finger_lo, finger_hi), 4)
    if position <= 2:
        return max(base_span - 1, 2)  # Tighter at low frets
    elif position >= 9:
        return base_span + 1  # Easier at high frets
    return base_span

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
    ('finger_cnn_ctx5_ft3.pth', 5, 0.4),  # ctx=5: 99.0% val (best pipeline w/ Phase 6)
    ('finger_cnn_ctx7_ft3.pth', 7, 0.6),  # ctx=7: 99.5% val (best pipeline w/ Phase 6)
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

    v8.3: Uses exhaustive search for small chords (≤4 fretted notes)
    to find the globally optimal finger assignment. Falls back to
    greedy for larger chords.

    Handles:
    1. Barre detection: same fret on 2+ strings → finger 1 (index barre)
    2. Exhaustive/greedy finger assignment for remaining notes
    3. Finger ordering: fret(I) ≤ fret(M) ≤ fret(R) ≤ fret(P)
    """
    if not chord_notes:
        return

    # --- Phase 1: Barre detection ---
    barres = _detect_barre(chord_notes)
    barre_notes = set()

    for barre_fret, barre_group in barres.items():
        for note in barre_group:
            note['left_hand_finger'] = 1
            barre_notes.add(id(note))

    # --- Phase 2: Assign non-barre notes ---
    non_barre = [n for n in chord_notes
                 if id(n) not in barre_notes and n.get('fret', 0) > 0]
    non_barre.sort(key=lambda n: n.get('fret', 0))

    if not non_barre:
        return

    # Determine position from barre fret or min fret
    if barres:
        position = min(barres.keys())
    else:
        fretted = [n.get('fret', 0) for n in chord_notes if n.get('fret', 0) > 0]
        position = min(fretted) if fretted else 1

    # Available fingers (1 may be used for barre)
    reserved = {1} if barres else set()
    available = [f for f in range(1, 5) if f not in reserved]

    if len(non_barre) <= 4 and len(non_barre) <= len(available):
        # --- Exhaustive search for small chords ---
        # Try all permutations of available fingers
        from itertools import permutations

        best_cost = float('inf')
        best_assignment = None

        for perm in permutations(available, len(non_barre)):
            cost = 0.0
            valid = True

            # Check finger ordering: fret(lower finger) ≤ fret(higher finger)
            fret_finger_pairs = []
            for note, finger in zip(non_barre, perm):
                fret = int(note.get('fret', 0))
                fret_finger_pairs.append((fret, finger))

            # Anatomical ordering check
            sorted_by_finger = sorted(fret_finger_pairs, key=lambda x: x[1])
            for k in range(1, len(sorted_by_finger)):
                if sorted_by_finger[k][0] < sorted_by_finger[k-1][0]:
                    valid = False
                    break

            if not valid:
                continue

            # Compute cost for this assignment
            for note, finger in zip(non_barre, perm):
                fret = int(note.get('fret', 0))
                offset = fret - position

                # Offset-based cost: standard offset=finger-1 is ideal
                if finger == offset + 1:
                    cost -= 4.0  # Strong bonus for standard offset
                elif 0 <= offset <= 3:
                    cost += abs(finger - (offset + 1)) * 2.0  # Penalty

                # CNN prior bonus
                probs = note.get('_finger_probs')
                if probs is not None:
                    cost -= 3.0 * float(probs[finger])

                # Span cost: fingers too far from position
                if offset < 0 or offset > 3:
                    cost += abs(offset - max(0, min(3, offset))) * 4.0

            # Inter-finger span cost within chord
            for i in range(len(non_barre)):
                for j in range(i + 1, len(non_barre)):
                    fi, fj = perm[i], perm[j]
                    fri, frj = int(non_barre[i].get('fret', 0)), int(non_barre[j].get('fret', 0))
                    lo, hi = min(fi, fj), max(fi, fj)
                    max_span = _BIO_MAX_SPAN.get((lo, hi), 4)
                    max_span = _position_adjusted_max_span(lo, hi, position)
                    actual_span = abs(fri - frj)
                    if actual_span > max_span:
                        cost += (actual_span - max_span) * 8.0

            if cost < best_cost:
                best_cost = cost
                best_assignment = perm

        if best_assignment:
            for note, finger in zip(non_barre, best_assignment):
                note['left_hand_finger'] = finger
        else:
            # Fallback to greedy if no valid permutation found
            _greedy_chord_assign(non_barre, position, reserved)
    else:
        # Greedy for larger chords
        _greedy_chord_assign(non_barre, position, reserved)

    # --- Phase 3: Enforce finger ordering ---
    _enforce_chord_finger_order(chord_notes)


def _greedy_chord_assign(non_barre, position, reserved):
    """Greedy fallback for chord finger assignment (>4 fretted notes)."""
    used_fingers = set(reserved)
    for note in non_barre:
        fret = note.get('fret', 0)
        if fret <= 0:
            continue
        offset = fret - position
        if 0 <= offset <= 3:
            ideal = offset + 1
        else:
            ideal = note.get('left_hand_finger', 2)
        if ideal not in used_fingers and 1 <= ideal <= 4:
            note['left_hand_finger'] = ideal
            used_fingers.add(ideal)
        else:
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


# ============================================================
# Viterbi DP Finger Assignment (v8)
# ============================================================

def _is_free_shift_point(phrase_notes: List[dict],
                         fretted_indices: List[int],
                         curr_idx: int) -> bool:
    """Check if a position shift can happen freely between two consecutive
    fretted notes at fretted_indices[curr_idx-1] and fretted_indices[curr_idx].

    Free shifts happen when:
    - An open string note exists between the two fretted notes
    - Time gap > 0.3 seconds between them
    - There's a rest (no note) > 0.2 seconds between them
    """
    if curr_idx <= 0 or curr_idx >= len(fretted_indices):
        return False

    prev_abs = fretted_indices[curr_idx - 1]
    curr_abs = fretted_indices[curr_idx]

    prev_note = phrase_notes[prev_abs]
    curr_note = phrase_notes[curr_abs]

    # Time gap > 0.3s → free shift
    time_gap = curr_note.get('start', 0) - prev_note.get('start', 0)
    if time_gap > 0.3:
        return True

    # Check for open strings between the two fretted notes
    for k in range(prev_abs + 1, curr_abs):
        if phrase_notes[k].get('fret', 0) == 0:
            return True

    # Check for rest (gap in note coverage) > 0.2s
    prev_end = prev_note.get('start', 0) + prev_note.get('duration', 0)
    curr_start = curr_note.get('start', 0)
    if curr_start - prev_end > 0.2:
        return True

    return False


def _finger_emission_cost(note: dict, finger: int, position: int) -> float:
    """Compute emission cost for assigning *finger* at *position* to *note*.

    Lower cost = better assignment.
    - CNN prior: -w_cnn_prior * cnn_prob[finger]
    - Derived rules: -w_offset_rule * offset_rule_prob[finger]
    - Standard offset bonus: -w_std_offset if finger == offset + 1
    - Anchor avoidance: penalty if finger is held by a sustained note
    - Barre context: bonus if position matches active barre
    - Chord position: bonus if position matches preceding chord
    """
    W = _ACTIVE_WEIGHTS
    cost = 0.0
    fret = int(note.get('fret', 0))
    offset = fret - position

    # CNN prior bonus (negative cost = reward)
    probs = note.get('_finger_probs')
    if probs is not None:
        cost -= W['w_cnn_prior'] * float(probs[finger])

    # Derived offset rules bonus
    _load_derived_rules()
    assert _FRET_OFFSET_RULES is not None
    offset_probs = _FRET_OFFSET_RULES.get(offset)
    if offset_probs and finger in offset_probs:
        cost -= W['w_offset_rule'] * offset_probs[finger]

    # Standard offset = finger - 1 bonus
    if finger == offset + 1:
        cost -= W['w_std_offset']
    else:
        # v8.3: Stretch penalty — non-standard offset costs more
        # This prevents the expanded state space from preferring stretch
        # positions when a standard position works equally well.
        cost += 6.0  # Moderate penalty for stretch states

    # --- v8.1: Anchor finger avoidance ---
    # If another sustained note is using this finger, avoid it
    # (Carlevaro "fijación": held fingers are immobilised)
    avoid_fingers = note.get('_avoid_fingers')
    if avoid_fingers and finger in avoid_fingers:
        cost += W['w_anchor_penalty']

    # --- v8.1: Barre context bonus ---
    # If the preceding chord established a barre, prefer staying in that position
    barre_ctx = note.get('_barre_context')
    if barre_ctx is not None and finger >= 2:
        barre_fret = int(barre_ctx)
        if position == barre_fret:
            cost += W['w_barre_continuity']  # Bonus (negative)

    # --- v8.1: Chord position context ---
    # Prefer staying in the position of the preceding chord
    chord_pos = note.get('_chord_position')
    if chord_pos is not None and position == int(chord_pos):
        cost += W['w_chord_pos_bonus']  # Bonus (negative)

    # --- v8.3: Technique-aware emission bias ---
    # Only activate when technique is explicitly provided (not 'normal')
    tech = note.get('_technique')
    if tech and tech != 'normal':
        if tech in ('slide_up', 'slide_down'):
            if finger <= 2:
                cost -= 3.0
        elif tech == 'bend':
            if finger in (2, 3):
                cost -= 4.0
        elif tech == 'harmonic':
            if finger == 1:
                cost -= 2.0

    # --- v8.3: String-based emission bias ---
    # Bass strings (4-6) favor thumb-side fingers for stability
    # Treble strings (1-2) favor pinky-side for melody agility
    # NOTE: Disabled — causes 1% solo regression without sufficient benefit
    # string = int(note.get('string', 3))
    # if string >= 5 and finger <= 2:  # Bass + index/middle
    #     cost -= 1.0
    # elif string <= 2 and finger >= 3:  # Treble + ring/pinky
    #     cost -= 0.5

    return cost


def _finger_transition_cost_dp(finger: int, prev_finger: int,
                               pos: int, prev_pos: int,
                               note: dict, prev_note: dict,
                               is_free_shift: bool) -> float:
    """Compute transition cost between consecutive finger states.

    Penalises / rewards:
    - Position shift distance (reduced if free shift)
    - Staying in same position (bonus)
    - Law 4 continuity (≤2 fret move bonus)
    - Finger crossing violation
    - Same finger on different fret
    - Span excess beyond bio-mechanical limits
    - Tendon coupling (ring + middle/pinky wide span)
    - Guide finger bonus (same string, same finger, different fret)
    """
    W = _ACTIVE_WEIGHTS
    cost = 0.0
    pos_diff = abs(pos - prev_pos)

    # --- Position shift cost ---
    if pos_diff == 0:
        cost += W['w_position_same']  # bonus (negative)
    else:
        shift_w = W['w_position_shift_free'] if is_free_shift else W['w_position_shift']
        cost += pos_diff * shift_w

    # --- Law 4: ≤2 fret move continuity bonus ---
    if pos_diff <= 2 and pos_diff > 0:
        cost += W['w_continuity_2fret']

    # --- Finger crossing violation ---
    fret = int(note.get('fret', 0))
    prev_fret = int(prev_note.get('fret', 0))
    if finger != prev_finger:
        fret_direction = fret - prev_fret  # positive = ascending
        finger_direction = finger - prev_finger  # positive = higher finger
        if fret_direction > 0 and finger_direction < 0:
            cost += W['w_finger_cross']
        elif fret_direction < 0 and finger_direction > 0:
            cost += W['w_finger_cross']

    # --- Same finger, different fret ---
    if finger == prev_finger and fret != prev_fret:
        # v8.3: Same finger is OK for slides (guide finger effect)
        tech = note.get('_technique')
        if tech and tech in ('slide_up', 'slide_down'):
            cost += W['w_guide_finger']  # Bonus instead of penalty
        else:
            cost += W['w_same_finger_diff']

    # --- Span excess (position-dependent, Miura 2003) ---
    if finger != prev_finger and pos == prev_pos:
        lo, hi = min(finger, prev_finger), max(finger, prev_finger)
        max_span = _position_adjusted_max_span(lo, hi, pos)
        actual_span = abs(fret - prev_fret)
        if actual_span > max_span:
            cost += (actual_span - max_span) * W['w_span_excess']

    # --- Tendon coupling: ring(3) + middle(2)/pinky(4) with wide span ---
    # Ref: Zatsiorsky et al. 2000 "enslaving effects in multi-finger
    #      force production" — ring finger has maximum tendon coupling
    if pos == prev_pos:
        pair = tuple(sorted([finger, prev_finger]))
        if pair in ((2, 3), (3, 4)):
            span = abs(fret - prev_fret)
            if span > 2:
                cost += W['w_tendon_coupling']

    # --- Guide finger bonus (same string, same finger, slide to new pos) ---
    # Ref: Segovia technique — guide finger stays on string during shifts
    if (finger == prev_finger and pos != prev_pos
            and note.get('string') == prev_note.get('string')):
        cost += W['w_guide_finger']

    # --- v8.2: String-crossing geometry ---
    # Large string jumps within the same position change hand geometry
    # significantly. Pressing string 1 (thinnest) while reaching to
    # string 6 (thickest) requires rotating the wrist.
    s = int(note.get('string', 3))
    prev_s = int(prev_note.get('string', 3))
    string_diff = abs(s - prev_s)
    if string_diff >= 3:
        cost += (string_diff - 2) * W.get('w_string_cross', 2.0)

    # --- v8.2: Bass/Treble voice crossing discount ---
    # In solo guitar, bass (strings 4-6) and melody (strings 1-3) are
    # semi-independent voices. Position shifts between voices should
    # be cheaper because the hand naturally adjusts when switching
    # between bass and treble register.
    if pos_diff > 0 and not is_free_shift:
        bass_to_treble = (prev_s >= 4 and s <= 3)
        treble_to_bass = (prev_s <= 3 and s >= 4)
        if bass_to_treble or treble_to_bass:
            # Reduce the position shift cost for voice crossings
            discount = W.get('w_voice_cross_discount', 0.5)
            cost -= pos_diff * shift_w * (1.0 - discount)

    # --- v8.2: Slide-based position shift bonus ---
    # When notes are on the same string and position shifts by 1-3 frets,
    # a slide technique can smooth the transition (cheaper than jumping).
    if (note.get('string') == prev_note.get('string')
            and pos != prev_pos and 1 <= pos_diff <= 3
            and finger != prev_finger):
        cost += W.get('w_slide_shift_bonus', -3.0)

    # --- v8.3: Sequential finger ordering bonus ---
    # Ascending fret with ascending finger (1→2→3→4) is ergonomic
    if fret > prev_fret and finger > prev_finger and pos == prev_pos:
        cost += W.get('w_finger_order', -2.0)
        # Adjacent finger pairs are even smoother
        if abs(finger - prev_finger) == 1:
            cost += W.get('w_finger_pair_smooth', -1.5)
    # Descending fret with descending finger (4→3→2→1) also ergonomic
    elif fret < prev_fret and finger < prev_finger and pos == prev_pos:
        cost += W.get('w_finger_order', -2.0)
        if abs(finger - prev_finger) == 1:
            cost += W.get('w_finger_pair_smooth', -1.5)

    # --- Minimax component (Hori & Sagayama 2016) ---
    # Prevent any single transition from being extremely difficult.
    # Instead of minimizing sum, penalize extreme single-step costs.
    minimax_thresh = W.get('w_minimax_threshold', 50.0)
    if cost > minimax_thresh:
        excess = cost - minimax_thresh
        cost += excess * W.get('w_minimax_excess', 3.0)

    return cost


def _viterbi_finger_phrase(fretted_notes: List[dict],
                           free_shift_set: set) -> int:
    """Run Viterbi DP on a sequence of fretted notes.

    For each note at fret F, valid states are (finger, position):
      - (1, F)   : index at fret F
      - (2, F-1) : middle at fret F  (pos >= 1)
      - (3, F-2) : ring at fret F    (pos >= 1)
      - (4, F-3) : pinky at fret F   (pos >= 1)

    Returns number of notes whose finger was changed.
    """
    N = len(fretted_notes)
    if N == 0:
        return 0

    def _states(note: dict) -> List[Tuple[int, int]]:
        """Return valid (finger, position) states for a note.

        v8.3: Standard 4 states per note.
        Stretch states available but gated behind per-note heuristic
        to avoid regression on solo passages.
        """
        fret = int(note.get('fret', 0))
        states = []
        seen = set()
        for finger in range(1, 5):
            pos = fret - (finger - 1)
            if pos >= 1 and (finger, pos) not in seen:
                states.append((finger, pos))
                seen.add((finger, pos))
        return states

    # --- Forward pass ---
    # dp[t] = { (finger, pos): (cumulative_cost, backpointer_state) }
    dp: List[dict] = []

    # Initialise t=0
    first_states = _states(fretted_notes[0])
    init: dict = {}
    for state in first_states:
        finger, pos = state
        cost = _finger_emission_cost(fretted_notes[0], finger, pos)
        init[state] = (cost, None)
    dp.append(init)

    # Fill t = 1 .. N-1
    for t in range(1, N):
        curr_states = _states(fretted_notes[t])
        is_free = t in free_shift_set
        curr_dp: dict = {}
        for state in curr_states:
            finger, pos = state
            em_cost = _finger_emission_cost(fretted_notes[t], finger, pos)
            best_cost = float('inf')
            best_prev = None
            for prev_state, (prev_cost, _) in dp[t - 1].items():
                prev_finger, prev_pos = prev_state
                tr_cost = _finger_transition_cost_dp(
                    finger, prev_finger, pos, prev_pos,
                    fretted_notes[t], fretted_notes[t - 1],
                    is_free)
                total = prev_cost + tr_cost + em_cost
                if total < best_cost:
                    best_cost = total
                    best_prev = prev_state
            curr_dp[state] = (best_cost, best_prev)
        dp.append(curr_dp)

    # --- Backtrack ---
    # Find best final state
    best_final_cost = float('inf')
    best_final_state = None
    for state, (cost, _) in dp[N - 1].items():
        if cost < best_final_cost:
            best_final_cost = cost
            best_final_state = state

    if best_final_state is None:
        return 0

    # Trace back
    path: List[Tuple[int, int]] = [best_final_state]
    for t in range(N - 1, 0, -1):
        _, prev_state = dp[t][path[-1]]
        if prev_state is None:
            break
        path.append(prev_state)
    path.reverse()

    # Apply results
    changed = 0
    for t, (finger, _pos) in enumerate(path):
        note = fretted_notes[t]
        old = note.get('left_hand_finger', 0)
        if old != finger:
            note['left_hand_finger'] = finger
            changed += 1

    return changed


def _compute_chord_ratio(phrase: List[dict]) -> float:
    """Compute what fraction of notes in a phrase are part of chords.

    Returns a value [0.0, 1.0] where 1.0 = all notes are simultaneous.
    Used for context-dependent weight blending (v8.3).
    """
    if len(phrase) <= 1:
        return 0.0
    simultaneous = 0
    for i in range(1, len(phrase)):
        gap = abs(phrase[i].get('start', 0) - phrase[i-1].get('start', 0))
        if gap <= 0.03:  # Same threshold as chord grouping
            simultaneous += 1
    return simultaneous / len(phrase)


def _blend_weights(chord_ratio: float) -> dict:
    """Blend solo and chord weights based on chord ratio.

    v8.3: Context-dependent weight selection.
    - chord_ratio = 0.0 → pure solo weights
    - chord_ratio = 1.0 → pure chord weights
    - Smooth interpolation in between

    Discovery from Optuna Phase 2: solo passages and chord voicings
    have fundamentally different optimal weight profiles.
    """
    if chord_ratio < 0.5:
        return _FINGER_DP_WEIGHTS
    if chord_ratio > 0.7:
        return _CHORD_DP_WEIGHTS

    # Linear blend
    blended = {}
    for key in _FINGER_DP_WEIGHTS:
        solo_val = _FINGER_DP_WEIGHTS[key]
        chord_val = _CHORD_DP_WEIGHTS.get(key, solo_val)
        blended[key] = solo_val + chord_ratio * (chord_val - solo_val)
    return blended


def _viterbi_finger_dp(notes: List[dict],
                       phrase_gap: float = 0.5) -> int:
    """Top-level Viterbi DP for finger assignment.

    v8.3: Context-dependent weight blending per phrase.

    1. Split notes into phrases by gap > phrase_gap
    2. Compute chord ratio per phrase
    3. Blend weights (solo ↔ chord) based on context
    4. Run _viterbi_finger_phrase with blended weights
    5. Return total number of notes changed.
    """
    global _ACTIVE_WEIGHTS
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    total_changed = 0

    # --- Split into phrases ---
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

    for phrase in phrases:
        # v8.3: Context-dependent weight selection
        chord_ratio = _compute_chord_ratio(phrase)
        _ACTIVE_WEIGHTS = _blend_weights(chord_ratio)

        # Identify fretted note indices within phrase
        fretted_indices = [
            i for i, n in enumerate(phrase)
            if isinstance(n.get('fret', 0), (int, float))
            and int(n.get('fret', 0)) > 0
        ]
        if len(fretted_indices) < 1:
            continue

        fretted_notes = [phrase[i] for i in fretted_indices]

        # Determine free shift points (indices within fretted_notes list)
        free_shift_set: set = set()
        for fi in range(1, len(fretted_indices)):
            if _is_free_shift_point(phrase, fretted_indices, fi):
                free_shift_set.add(fi)

        changed = _viterbi_finger_phrase(fretted_notes, free_shift_set)
        total_changed += changed

    # Restore default weights
    _ACTIVE_WEIGHTS = _FINGER_DP_WEIGHTS

    # --- v8.2: Cross-phrase transition optimization ---
    total_changed += _optimize_phrase_transitions(phrases)

    return total_changed


def _optimize_phrase_transitions(phrases: List[List[dict]]) -> int:
    """Optimize fingering at phrase boundaries for smoother transitions.

    Human guitarists anticipate upcoming position shifts during the
    current phrase. Instead of an abrupt jump at the phrase boundary,
    they gradually shift the last 2-3 notes toward the target position.

    Ref: Radicioni & Lombardo (2005) - "solve per segment, optimize
         transitions between segments"
    Ref: Tennant - "anticipation/visualization: mentally prepare for
         next chord/note before moving"

    Returns number of notes adjusted.
    """
    if len(phrases) < 2:
        return 0

    adjusted = 0

    for pi in range(len(phrases) - 1):
        curr_phrase = phrases[pi]
        next_phrase = phrases[pi + 1]

        # Get fretted notes from each phrase
        curr_fretted = [n for n in curr_phrase
                        if isinstance(n.get('fret', 0), (int, float))
                        and int(n.get('fret', 0)) > 0]
        next_fretted = [n for n in next_phrase
                        if isinstance(n.get('fret', 0), (int, float))
                        and int(n.get('fret', 0)) > 0]

        if not curr_fretted or not next_fretted:
            continue

        # Estimate positions
        curr_pos = _estimate_position(curr_fretted[-3:]) if len(curr_fretted) >= 3 \
            else _estimate_position(curr_fretted)
        next_pos = _estimate_position(next_fretted[:3]) if len(next_fretted) >= 3 \
            else _estimate_position(next_fretted)

        if curr_pos is None or next_pos is None:
            continue

        pos_jump = abs(next_pos - curr_pos)
        if pos_jump <= 2:
            # Small jump — no preparation needed
            continue

        # Adjust the LAST 2-3 notes of the current phrase
        # Try to shift them toward the target position
        tail_count = min(3, len(curr_fretted))
        tail_notes = curr_fretted[-tail_count:]

        for note in tail_notes:
            fret = int(note.get('fret', 0))
            old_finger = note.get('left_hand_finger', 0)
            if fret <= 0 or old_finger <= 0:
                continue

            old_pos = fret - (old_finger - 1)

            # Try each finger to see if one brings position closer to target
            best_finger = old_finger
            best_distance = abs(old_pos - next_pos)

            for candidate in range(1, 5):
                cand_pos = fret - (candidate - 1)
                if cand_pos < 1:
                    continue
                if not _is_valid_finger(fret, candidate):
                    continue
                cand_dist = abs(cand_pos - next_pos)
                if cand_dist < best_distance:
                    best_finger = candidate
                    best_distance = cand_dist

            if best_finger != old_finger:
                note['left_hand_finger'] = best_finger
                adjusted += 1

    return adjusted



def _apply_pitch_proximity_rule(notes: List[dict]) -> int:
    """Step 3.5: Law 3 — Pitch proximity preserves position.

    When consecutive notes are < 3 semitones apart and on the same string,
    they should use the same hand position. This prevents unnecessary
    position changes for chromatic or whole-step movements.

    Statistics (from 8.1M notes):
      0 semitones: 96.6% same-string
      1 semitone:  76.6% same-string
      2 semitones: 63.0% same-string
      3 semitones: 18.7% same-string (boundary)

    Returns number of notes corrected."""
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    corrected = 0

    for i in range(1, len(sorted_notes)):
        prev = sorted_notes[i - 1]
        curr = sorted_notes[i]

        prev_fret = int(prev.get('fret', 0))
        curr_fret = int(curr.get('fret', 0))
        if prev_fret <= 0 or curr_fret <= 0:
            continue

        prev_pitch = int(prev.get('pitch', 0))
        curr_pitch = int(curr.get('pitch', 0))
        interval = abs(curr_pitch - prev_pitch)

        # Only apply when pitch interval < 3 semitones
        if interval >= 3:
            continue

        # Check if on same string
        if prev.get('string') != curr.get('string'):
            continue

        # Skip if time gap is large (different phrase)
        time_gap = curr.get('start', 0) - prev.get('start', 0)
        if time_gap > 0.5:
            continue

        prev_finger = int(prev.get('left_hand_finger', 0))
        if prev_finger <= 0:
            continue

        # Compute prev note's position
        prev_pos = prev_fret - (prev_finger - 1)
        if prev_pos < 1:
            continue

        # Check if curr note fits in the same position
        curr_offset = curr_fret - prev_pos
        if 0 <= curr_offset <= 3:
            ideal_finger = curr_offset + 1
            curr_finger = int(curr.get('left_hand_finger', 0))
            if curr_finger != ideal_finger and _is_valid_finger(curr_fret, ideal_finger):
                # Within same position, offset-based fingers are always
                # anatomically valid (fret order = finger order by construction)
                curr['left_hand_finger'] = ideal_finger
                corrected += 1

    return corrected


def _enforce_pattern_consistency(notes: List[dict],
                                 min_pattern_len: int = 4) -> int:
    """Detect repeated pitch patterns and ensure consistent fingering.

    Uses MAJORITY VOTE across all occurrences of a repeated pitch pattern:
    1. Create sliding windows of pitch sequences (length 4-8)
    2. Hash each window by (pitch_tuple, string_tuple)
    3. Collect all finger assignments for each pattern
    4. Compute majority (most common) finger for each position
    5. Apply majority fingering to ALL occurrences

    This prevents a bad first-occurrence from propagating its errors
    to all subsequent repetitions.

    Returns number of notes corrected.
    """
    from collections import Counter

    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    # Only consider fretted notes for pattern matching
    fretted = [n for n in sorted_notes
               if isinstance(n.get('fret', 0), (int, float))
               and int(n.get('fret', 0)) > 0]
    if len(fretted) < min_pattern_len:
        return 0

    corrected = 0

    for win_len in range(min_pattern_len, min(9, len(fretted) + 1)):
        # Phase 1: Collect all occurrences of each pattern
        # Key: (pitch_tuple, string_tuple) -> list of (start_index, finger_tuple)
        pattern_occurrences: dict = {}
        for i in range(len(fretted) - win_len + 1):
            window = fretted[i:i + win_len]
            pitch_key = tuple(int(n.get('pitch', 0)) for n in window)
            string_key = tuple(int(n.get('string', 0)) for n in window)
            finger_key = tuple(int(n.get('left_hand_finger', 0)) for n in window)
            pattern_key = (pitch_key, string_key)

            if pattern_key not in pattern_occurrences:
                pattern_occurrences[pattern_key] = []
            pattern_occurrences[pattern_key].append((i, finger_key))

        # Phase 2: For patterns with multiple occurrences, compute majority vote
        for pattern_key, occurrences in pattern_occurrences.items():
            if len(occurrences) < 2:
                continue

            # Compute majority finger for each position in the window
            majority_fingers = []
            for pos in range(win_len):
                finger_counts = Counter(
                    occ_fingers[pos]
                    for _, occ_fingers in occurrences
                    if occ_fingers[pos] > 0  # Skip unassigned
                )
                if finger_counts:
                    majority_fingers.append(finger_counts.most_common(1)[0][0])
                else:
                    majority_fingers.append(0)

            majority_tuple = tuple(majority_fingers)

            # Phase 3: Apply majority fingering to all occurrences
            for start_idx, occ_fingers in occurrences:
                if occ_fingers == majority_tuple:
                    continue  # Already matches majority
                window = fretted[start_idx:start_idx + win_len]
                for j in range(win_len):
                    old_finger = int(window[j].get('left_hand_finger', 0))
                    new_finger = majority_fingers[j]
                    if old_finger != new_finger and new_finger > 0:
                        window[j]['left_hand_finger'] = new_finger
                        corrected += 1

    return corrected


def _apply_pivot_fingers(notes: List[dict]) -> int:
    """For chord-to-chord transitions, keep common (string, fret) on the
    same finger.

    1. Identify consecutive chord groups (simultaneous notes)
    2. Find common (string, fret) pairs between adjacent chords
    3. Ensure the common notes keep the same finger

    Returns number of fixes.
    """
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    if not sorted_notes:
        return 0

    # Group into chords (simultaneous notes within 0.03s)
    groups: List[List[dict]] = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if note.get('start', 0) - groups[-1][0].get('start', 0) <= 0.03:
            groups[-1].append(note)
        else:
            groups.append([note])

    # Only consider groups with multiple notes (actual chords)
    chord_groups = [g for g in groups if len(g) > 1]
    if len(chord_groups) < 2:
        return 0

    fixes = 0
    for ci in range(1, len(chord_groups)):
        prev_chord = chord_groups[ci - 1]
        curr_chord = chord_groups[ci]

        # Build lookup: (string, fret) -> note for previous chord
        prev_map: dict = {}
        for n in prev_chord:
            sf = (n.get('string'), n.get('fret'))
            if sf[1] is not None and int(sf[1]) > 0:
                prev_map[sf] = n

        # Check current chord notes for common (string, fret)
        for n in curr_chord:
            sf = (n.get('string'), n.get('fret'))
            if sf in prev_map:
                prev_note = prev_map[sf]
                prev_finger = prev_note.get('left_hand_finger', 0)
                curr_finger = n.get('left_hand_finger', 0)
                if (prev_finger > 0 and curr_finger != prev_finger
                        and _is_valid_finger(int(sf[1]), prev_finger)):
                    n['left_hand_finger'] = prev_finger
                    fixes += 1

    return fixes


def assign_fingers(notes: List[dict], phrase_gap: float = 0.5,
                    techniques: Optional[List[str]] = None,
                    detected_key: Optional[str] = None) -> List[dict]:
    """Main API: Assign left_hand_finger (0-4) to each note.
    CNN-first with biomechanical post-processing.

    Pipeline v8.1: Paper-informed improvements
      - Anchor finger detection (Carlevaro fijación)
      - Barre context propagation (barre as position approach)
      - Chord position persistence (hold chord shape for melody)
      - Position-dependent span (Miura 2003)
      - Near-hard finger crossing constraint (Radicioni CSP)
      - Minimax cost component (Hori & Sagayama 2016)

    Args:
        notes: List of note dicts with string, fret, pitch, start keys
        phrase_gap: Gap in seconds to split phrases for smoothing
        techniques: Optional list of technique strings (1:1 with notes).
                    Values: 'normal', 'hammer_on', 'pull_off', 'slide_up',
                    'slide_down', 'bend', 'harmonic', etc.
        detected_key: Optional key signature (e.g. 'Am', 'C', 'G')
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

    # Step 2.5 (v8.1): Context propagation — inform Viterbi DP
    anchor_count = _mark_anchor_context(notes)
    barre_count = _propagate_barre_context(notes, groups)
    chord_pos_count = _propagate_chord_position(notes, groups)

    # Step 3: Viterbi DP finger assignment (replaces position_smoothing + scale runs)
    viterbi_fixes = _viterbi_finger_dp(notes, phrase_gap=phrase_gap)

    # Note: Post-Viterbi chord re-resolution was tried but caused -12.2%
    # regression by overwriting Viterbi's globally-optimal assignments.
    # Viterbi handles chord notes adequately through context-dependent weights.
    chord_refix = 0

    # Step 3.5: Law 3 — Pitch proximity preserves position
    prox_fixes = _apply_pitch_proximity_rule(notes)

    # Step 4: Pattern consistency
    pattern_fixes = _enforce_pattern_consistency(notes)

    # Step 4.5: Pivot fingers for chord transitions
    pivot_fixes = _apply_pivot_fingers(notes)

    # Step 5: Technique-aware finger constraints
    tech_fixes = _apply_technique_constraints(notes)

    # Cleanup temp attributes
    for note in notes:
        note.pop('_finger_conf', None)
        note.pop('_finger_probs', None)
        note.pop('_technique', None)
        note.pop('_avoid_fingers', None)
        note.pop('_barre_context', None)
        note.pop('_chord_position', None)

    mode = "CNN" if use_cnn else "PDMX"
    print(f"[finger_assigner] {len(notes)} notes ({mode}, "
          f"{len(groups)} groups, viterbi={viterbi_fixes}, prox={prox_fixes}, "
          f"pattern={pattern_fixes}, pivot={pivot_fixes}, tech={tech_fixes}, "
          f"anchor={anchor_count}, barre_ctx={barre_count}, chord_ctx={chord_pos_count})")
    return notes


def _mark_anchor_context(notes: List[dict]) -> int:
    """Detect sustained notes whose fingers are 'anchored' (held down).

    When a note's duration extends past the start of subsequent notes,
    those subsequent notes should NOT reuse the anchor finger.

    Ref: Carlevaro 'fijación' — controlled immobility of specific fingers.
    Ref: Tennant — finger preparation requires keeping held fingers still.

    Sets note['_avoid_fingers'] = set of fingers to avoid.
    Returns number of notes tagged.
    """
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    tagged = 0

    for i, note in enumerate(sorted_notes):
        start = float(note.get('start', 0))
        duration = float(note.get('duration', note.get('end', start) - start))
        if duration <= 0:
            # Estimate duration from next note's start
            if i + 1 < len(sorted_notes):
                duration = float(sorted_notes[i + 1].get('start', start)) - start
            else:
                continue
        end = start + duration
        fret = note.get('fret', 0)
        finger = note.get('left_hand_finger', 0)

        if not isinstance(fret, (int, float)) or fret <= 0 or finger <= 0:
            continue

        # Check if this note sustains past subsequent notes
        for j in range(i + 1, min(i + 6, len(sorted_notes))):
            next_note = sorted_notes[j]
            next_start = float(next_note.get('start', 0))
            if next_start >= end:
                break
            next_fret = next_note.get('fret', 0)
            if not isinstance(next_fret, (int, float)) or next_fret <= 0:
                continue
            # This note is still sounding — mark finger as unavailable
            if '_avoid_fingers' not in next_note:
                next_note['_avoid_fingers'] = set()
            next_note['_avoid_fingers'].add(finger)
            tagged += 1

    return tagged


def _estimate_tempo(notes: List[dict]) -> float:
    """Estimate tempo (BPM) from note durations/inter-onset intervals.

    Uses median inter-onset interval of fretted notes to approximate
    the beat period. Returns estimated BPM, or 120.0 (default) when
    there is insufficient data.
    """
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    fretted = [n for n in sorted_notes
               if isinstance(n.get('fret', 0), (int, float))
               and int(n.get('fret', 0)) > 0]
    if len(fretted) < 4:
        return 120.0

    iois = []
    for i in range(1, len(fretted)):
        gap = fretted[i].get('start', 0) - fretted[i - 1].get('start', 0)
        if 0.05 < gap < 2.0:  # Filter out simultaneous notes and long rests
            iois.append(gap)

    if len(iois) < 3:
        return 120.0

    iois.sort()
    median_ioi = iois[len(iois) // 2]
    if median_ioi <= 0:
        return 120.0

    # Assume median IOI ≈ one beat
    estimated_bpm = 60.0 / median_ioi
    # Clamp to reasonable range
    return max(40.0, min(240.0, estimated_bpm))


def _propagate_barre_context(notes: List[dict],
                              groups: List[List[dict]]) -> int:
    """Propagate barre chord context to subsequent single notes.

    When a chord has a barre (finger 1 covering multiple strings at the
    same fret), subsequent notes should prefer staying in that position.
    This models how guitarists use barre as a 'position anchor' — the
    index finger stays barred while other fingers move around it.

    'If the next note is far, barre to get close.'
    — This allows the hand to approach distant notes by establishing
      a barre position first, then using fingers 2-4 for melody.

    The context window adapts to tempo: at slower tempos the window is
    wider (more time between notes), at faster tempos it shrinks.
    Base window is 2.0s at 120 BPM.

    Tags subsequent notes with '_barre_context' = barre_fret.
    Returns number of notes tagged.
    """
    tagged = 0

    # Tempo-adaptive window: base 2.0s at 120 BPM
    all_notes = [n for g in groups for n in g]
    tempo = _estimate_tempo(all_notes)
    base_window = 2.0
    barre_window = base_window * (120.0 / tempo)

    # Find chord groups with barre
    for gi, group in enumerate(groups):
        if len(group) < 2:
            continue

        # Check if this group has a barre (finger 1 on 2+ strings at same fret)
        finger1_frets = [n.get('fret', 0) for n in group
                         if n.get('left_hand_finger', 0) == 1
                         and n.get('fret', 0) > 0]
        if len(finger1_frets) < 2:
            continue

        from collections import Counter
        fret_counts = Counter(finger1_frets)
        barre_fret = fret_counts.most_common(1)[0][0]
        if fret_counts[barre_fret] < 2:
            continue

        # Barre detected at barre_fret
        chord_end_time = max(float(n.get('start', 0)) for n in group) + 0.05

        # Tag subsequent notes within tempo-adaptive window (or until next chord group)
        next_chord_time = float('inf')
        for gj in range(gi + 1, len(groups)):
            if len(groups[gj]) > 1:
                next_chord_time = float(groups[gj][0].get('start', float('inf')))
                break

        max_time = min(chord_end_time + barre_window, next_chord_time)

        for gj in range(gi + 1, len(groups)):
            for note in groups[gj]:
                t = float(note.get('start', 0))
                if t >= max_time:
                    break
                fret = note.get('fret', 0)
                if isinstance(fret, (int, float)) and fret > 0:
                    note['_barre_context'] = barre_fret
                    tagged += 1

    return tagged


def _propagate_chord_position(notes: List[dict],
                               groups: List[List[dict]]) -> int:
    """Propagate chord hand position to subsequent single notes.

    Solo guitarists hold chord shapes and add/lift individual fingers
    for melody notes. The chord shape provides a 'home base'.

    After a chord group, single notes within a tempo-adaptive window
    get tagged with the chord's position, giving them a Viterbi DP
    bonus for staying in position. Base window is 1.0s at 120 BPM.

    Tags notes with '_chord_position' = estimated position.
    Returns number of notes tagged.
    """
    tagged = 0

    # Tempo-adaptive window: base 1.0s at 120 BPM
    all_notes = [n for g in groups for n in g]
    tempo = _estimate_tempo(all_notes)
    base_window = 1.0
    chord_window = base_window * (120.0 / tempo)

    for gi, group in enumerate(groups):
        if len(group) < 2:
            continue

        # Estimate chord position (lowest fretted note's position)
        fretted = [n for n in group
                   if isinstance(n.get('fret', 0), (int, float))
                   and n.get('fret', 0) > 0]
        if not fretted:
            continue

        chord_pos = _estimate_position(fretted)
        if chord_pos is None or chord_pos < 1:
            continue

        chord_end_time = max(float(n.get('start', 0)) for n in group) + 0.05

        # Tag subsequent single notes within tempo-adaptive window
        next_chord_time = float('inf')
        for gj in range(gi + 1, len(groups)):
            if len(groups[gj]) > 1:
                next_chord_time = float(groups[gj][0].get('start', float('inf')))
                break

        max_time = min(chord_end_time + chord_window, next_chord_time)

        for gj in range(gi + 1, len(groups)):
            for note in groups[gj]:
                t = float(note.get('start', 0))
                if t >= max_time:
                    break
                fret = note.get('fret', 0)
                if isinstance(fret, (int, float)) and fret > 0:
                    note['_chord_position'] = chord_pos
                    tagged += 1

    return tagged


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

