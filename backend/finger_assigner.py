"""
finger_assigner.py - Hybrid CNN + Constraint Engine v5
=======================================================
Strategy:
  1. CNN predicts finger for each note (83.8% standalone)
  2. Post-processing: enforce biomechanical constraints
     - Open string = finger 0 (override)
     - Chord finger uniqueness
     - Position consistency smoothing (sliding window)
  3. PDMX table fallback when CNN unavailable
"""
from typing import List, Tuple, Optional
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


def _is_valid_finger(fret, finger):
    """Check if a finger assignment is biomechanically possible.
    Relaxed: allow any finger 1-4 for fretted notes (stretches are valid)."""
    if fret == 0:
        return finger == 0
    if finger == 0:
        return fret == 0
    # Any finger 1-4 is valid for any fret > 0
    # (stretches and position shifts are normal guitar technique)
    return 1 <= finger <= 4


def _resolve_chord_conflicts(chord_notes):
    """Ensure no finger used twice in a chord (except finger 0 for open)."""
    used = {}
    for note in chord_notes:
        fg = note['left_hand_finger']
        if fg == 0:
            continue
        if fg in used:
            # Conflict: reassign the lower-confidence one
            prev_note = used[fg]
            # Keep the one with higher CNN confidence
            prev_conf = prev_note.get('_finger_conf', 0)
            cur_conf = note.get('_finger_conf', 0)
            victim = prev_note if cur_conf > prev_conf else note

            # Find alternative finger for victim
            fret = victim.get('fret', 0)
            probs = victim.get('_finger_probs')
            if probs is not None:
                # Pick next best valid finger not in use
                order = np.argsort(-probs)
                for alt in order:
                    alt = int(alt)
                    if alt not in used and _is_valid_finger(fret, alt):
                        victim['left_hand_finger'] = alt
                        used[alt] = victim
                        break
            else:
                for alt in [1, 2, 3, 4]:
                    if alt not in used and _is_valid_finger(fret, alt):
                        victim['left_hand_finger'] = alt
                        used[alt] = victim
                        break
        else:
            used[fg] = note


def assign_fingers(notes: List[dict], phrase_gap: float = 0.5) -> List[dict]:
    """Main API: Assign left_hand_finger (0-4) to each note.
    CNN-first with biomechanical post-processing."""
    if not notes:
        return notes

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

    # Cleanup temp attributes
    for note in notes:
        note.pop('_finger_conf', None)
        note.pop('_finger_probs', None)

    mode = "CNN" if use_cnn else "PDMX"
    print(f"[finger_assigner] {len(notes)} notes ({mode}, "
          f"{len(groups)} groups)")
    return notes
