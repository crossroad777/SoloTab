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
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import math
import os
import sys
from typing import List, Tuple, Optional, Dict
import numpy as np
from fingering_template_db import apply_phrase_templates

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
    'w_cnn_prior': 34.99,
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
    'w_anchor_penalty': 60.0,
    'w_chord_pos_bonus': -23.47,
    'w_string_cross': 9.22,
    'w_voice_cross_discount': 0.83,
    'w_slide_shift_bonus': -5.76,
    'w_pivot_finger': -15.0,
    'w_descending_shift_factor': 1.3,
    'w_stretch_penalty_base': 6.0,
    'w_lh_shift_rh_repeat_penalty': 2.0,
    'w_lh_pinky_rh_thumb_bass_penalty': 3.0,
    'w_lh_pivot_rh_alternation_bonus': -1.5,
    'w_presentacion_lookahead': 2.0,
    'w_tech_slide_bonus': -3.0,
    'w_tech_bend_bonus': -4.0,
    'w_tech_vibrato_pinky_penalty': 5.0,
    'w_tech_harmonic_bonus': -2.0,
    'w_tech_hammer_pull_bonus': -4.0,
    'w_bend_support_conflict_penalty': 4.0,
    'w_bass_sustain_bonus': -3.0,
    'w_lh_fatigue_penalty': 2.0,
    'w_wrist_angle_penalty': 3.0,
    'w_finger_order': 0.0,
    'w_finger_pair_smooth': 0.0,
    'w_scale_box_bonus': -10.0,
    'w_common_tone_bonus': -4.0,
    'w_conjunct_bass_bonus': -3.0,
    'w_disjunct_bass_penalty': 5.0,
    'w_data_prior': -1.5,
    'w_string_finger_prior': 3.0,
}

# Chord weights from Optuna Phase 6
_CHORD_DP_WEIGHTS = {
    'w_cnn_prior': 10.4000,
    'w_offset_rule': 20.0000,
    'w_std_offset': 3.4500,
    'w_position_same': -7.2500,
    'w_position_shift': 15.3300,
    'w_position_shift_free': 2.4100,
    'w_finger_cross': 200.0000,
    'w_same_finger_diff': 23.9100,
    'w_span_excess': 16.7000,
    'w_tendon_coupling': 10.0000,
    'w_continuity_2fret': -7.6700,
    'w_guide_finger': -24.3700,
    'w_minimax_threshold': 87.7500,
    'w_minimax_excess': 1.6400,
    'w_barre_continuity': -16.9900,
    'w_anchor_penalty': 60.0,
    'w_chord_pos_bonus': -17.9800,
    'w_string_cross': 1.3600,
    'w_voice_cross_discount': 0.3900,
    'w_slide_shift_bonus': -12.8200,
    'w_finger_order': 0.0,
    'w_finger_pair_smooth': 0.0,
    'w_pivot_finger': -15.0000,
    'w_descending_shift_factor': 1.3000,
    'w_stretch_penalty_base': 6.0000,
    'w_lh_shift_rh_repeat_penalty': 2.0000,
    'w_lh_pinky_rh_thumb_bass_penalty': 3.0000,
    'w_lh_pivot_rh_alternation_bonus': -1.5000,
    'w_presentacion_lookahead': 2.0000,
    'w_tech_slide_bonus': -3.0,
    'w_tech_bend_bonus': -4.0,
    'w_tech_vibrato_pinky_penalty': 5.0,
    'w_tech_harmonic_bonus': -2.0,
    'w_tech_hammer_pull_bonus': -4.0,
    'w_bend_support_conflict_penalty': 4.0,
    'w_bass_sustain_bonus': -3.0,
    'w_lh_fatigue_penalty': 2.0,
    'w_wrist_angle_penalty': 3.0,
    'w_scale_box_bonus': -5.0,
    'w_common_tone_bonus': -2.0,
    'w_conjunct_bass_bonus': -2.0,
    'w_disjunct_bass_penalty': 3.0,
    'w_data_prior': -0.5,
    'w_string_finger_prior': 2.0,
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
_STRING_FINGER_RULES = None  # string → {finger: prob}


def _load_derived_rules():
    """Load fret_offset_rules and string_finger_usage from derived_fingering_rules.json."""
    global _FRET_OFFSET_RULES, _STRING_FINGER_RULES
    if _FRET_OFFSET_RULES is not None and _STRING_FINGER_RULES is not None:
        return
    rules_path = os.path.join(os.path.dirname(__file__),
                              'derived_fingering_rules.json')
    if os.path.exists(rules_path):
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # fret_offset_rules
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

            # string_finger_usage
            raw_s = data.get('string_finger_usage', {})
            _STRING_FINGER_RULES = {}
            for s_str, f_counts in raw_s.items():
                total = sum(f_counts.values())
                if total > 0:
                    _STRING_FINGER_RULES[int(s_str)] = {
                        int(fg): cnt / total
                        for fg, cnt in f_counts.items()
                    }
        except Exception as e:
            print(f"[finger_assigner] derived rules load failed: {e}")
            _FRET_OFFSET_RULES = {}
            _STRING_FINGER_RULES = {}
    else:
        _FRET_OFFSET_RULES = {}
        _STRING_FINGER_RULES = {}


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
# Mined Fingering Patterns (from GP5 corpus 4.8M notes)
# ============================================================
_MINED_RUN2_PATTERNS = None


def _load_mined_patterns():
    """Load scale_run2_fingerings from mined_fingering_patterns.json."""
    global _MINED_RUN2_PATTERNS
    if _MINED_RUN2_PATTERNS is not None:
        return _MINED_RUN2_PATTERNS
    path_candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'gp_training_data', 'mined_fingering_patterns.json'),
        os.path.join(os.path.dirname(__file__), 'gp_training_data', 'mined_fingering_patterns.json'),
    ]
    db_path = None
    for p in path_candidates:
        if os.path.exists(p):
            db_path = p
            break
    if db_path and os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _MINED_RUN2_PATTERNS = data.get('scale_run2_fingerings', {})
        except Exception as e:
            print(f"[finger_assigner] mined patterns load failed: {e}")
            _MINED_RUN2_PATTERNS = {}
    else:
        _MINED_RUN2_PATTERNS = {}
    return _MINED_RUN2_PATTERNS


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
    Returns dict of {fret: [notes]} for frets with 2+ notes.
    修正: 2音のみのバレーは弦間距離が近い場合のみ認める。
    """
    from collections import defaultdict
    fret_groups = defaultdict(list)
    for note in chord_notes:
        fret = note.get('fret', 0)
        if fret > 0:
            fret_groups[fret].append(note)

    result = {}
    for f, notes in fret_groups.items():
        if len(notes) >= 3:
            # 3音以上なら無条件でバレー
            result[f] = notes
        elif len(notes) == 2:
            # 2音のみのバレーは弦間距離が2以内の場合のみ
            strings = sorted(int(n.get('string', 0)) for n in notes)
            span = max(strings) - min(strings)
            if span <= 2:
                result[f] = notes
            # span > 2 の場合はバレーとしない（個別の指を割り当てる）
    return result


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

            # Left-Hand Cross-String Crossing Constraint
            for i in range(len(non_barre)):
                si = int(non_barre[i].get('string', 0))
                fi = perm[i]
                fri = int(non_barre[i].get('fret', 0))
                for j in range(i + 1, len(non_barre)):
                    sj = int(non_barre[j].get('string', 0))
                    fj = perm[j]
                    frj = int(non_barre[j].get('fret', 0))

                    # Reject if si > sj, fi < fj, fri > frj (crossed finger hand contortion)
                    if si > sj and fi < fj and fri > frj:
                        valid = False
                        break
                    if sj > si and fj < fi and frj > fri:
                        valid = False
                        break

                    # Or same fret and non-adjacent fingers
                    if fri == frj and si != sj:
                        if abs(fi - fj) > 1:
                            valid = False
                            break
                if not valid:
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

                # v18.0: コード内の押弦における指の疲労度と手首ねじれコストの反映
                string = int(note.get('string', 0))
                if string >= 5:  # 低音弦（5・6弦）
                    # 疲労ペナルティ
                    if finger == 4:
                        cost += _CHORD_DP_WEIGHTS.get('w_lh_fatigue_penalty', 2.0) * 1.5
                    elif finger == 3:
                        cost += _CHORD_DP_WEIGHTS.get('w_lh_fatigue_penalty', 2.0) * 0.8
                    
                    # 手首ねじれペナルティ
                    if finger == 4 and fret >= 8:
                        cost += _CHORD_DP_WEIGHTS.get('w_wrist_angle_penalty', 3.0)

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
            if time_gap < 0.5:
                continue
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
        # v25.0 (Approach A): Clip CNN probability to prevent it from completely
        # overriding biomechanical constraints.
        prob_val = min(0.70, float(probs[finger]))
        cost -= W['w_cnn_prior'] * prob_val

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
        # v8.5: Position-dependent stretch penalty (Radicioni 2004)
        # High positions have narrower fret spacing → stretch is easier.
        # Restrict reduction to high positions (fret >= 9) to prevent mid-position stretch abuse.
        fret_val = int(note.get('fret', 0))
        if fret_val >= 9:
            pos_factor = max(0.4, 1.0 - fret_val * 0.05)
        else:
            pos_factor = 1.0
        cost += W.get('w_stretch_penalty_base', 6.0) * pos_factor

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

    # --- v24.0: Scale box match bonus ---
    scale_finger = note.get('_scale_finger')
    if scale_finger is not None and finger == scale_finger:
        cost += W.get('w_scale_box_bonus', -10.0)

    # --- v24.0: String-finger usage prior ---
    _load_derived_rules()
    if _STRING_FINGER_RULES:
        s = int(note.get('string', 3))
        s_probs = _STRING_FINGER_RULES.get(s)
        if s_probs and finger in s_probs:
            cost -= W.get('w_string_finger_prior', 3.0) * s_probs[finger]

    # --- v13.0: Technique-aware emission bias ---
    # Only activate when technique is explicitly provided (not 'normal')
    tech = note.get('_technique') or note.get('technique')
    if (tech and tech != 'normal') or note.get('vibrato') or note.get('_vibrato'):
        if tech in ('slide_up', 'slide_down'):
            if finger <= 2:
                cost += W.get('w_tech_slide_bonus', -3.0)
        elif tech == 'bend':
            if finger in (2, 3):
                cost += W.get('w_tech_bend_bonus', -4.0)

            # Support finger conflict penalty (fijación de apoio)
            occupied = note.get('_chord_occupied_fingers', set())
            if finger == 3 and (1 in occupied or 2 in occupied):
                cost += W.get('w_bend_support_conflict_penalty', 4.0)
            elif finger == 4 and (2 in occupied or 3 in occupied):
                cost += W.get('w_bend_support_conflict_penalty', 4.0)
            elif finger == 2 and (1 in occupied):
                cost += W.get('w_bend_support_conflict_penalty', 4.0)
        elif tech == 'vibrato' or note.get('vibrato') or note.get('_vibrato'):
            if finger == 4:
                cost += W.get('w_tech_vibrato_pinky_penalty', 5.0)
            elif finger in (2, 3):
                cost += W.get('w_tech_bend_bonus', -4.0)
        elif tech == 'harmonic':
            if finger == 1:
                cost += W.get('w_tech_harmonic_bonus', -2.0)

    # --- v8.3: String-based emission bias ---
    # Bass strings (4-6) favor thumb-side fingers for stability
    # Treble strings (1-2) favor pinky-side for melody agility
    # NOTE: Disabled — causes 1% solo regression without sufficient benefit
    # string = int(note.get('string', 3))
    # if string >= 5 and finger <= 2:  # Bass + index/middle
    #     cost -= 1.0
    # elif string <= 2 and finger >= 3:  # Treble + ring/pinky
    #     cost -= 0.5

    # --- v16.0: Finger Fatigue & Wrist Twist Penalty ---
    string = int(note.get('string', 3))
    if string >= 5:  # 低音弦（5・6弦）
        # 疲労ペナルティ: 弱い指（薬指3、小指4）への負荷を抑える
        if finger == 4:
            cost += W.get('w_lh_fatigue_penalty', 2.0) * 1.5
        elif finger == 3:
            cost += W.get('w_lh_fatigue_penalty', 2.0) * 0.8
        
        # 手首ねじれペナルティ: 低音弦で高いフレットを小指で押さえる無理な角度を回避
        if finger == 4 and fret >= 8:
            cost += W.get('w_wrist_angle_penalty', 3.0)

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
    tech = note.get('_technique') or note.get('technique')

    # v9.0: Tempo-adaptive guide finger slide IOI threshold
    # v25.0 (Approach A): Impose a physical lower bound of 0.25s on guide finger threshold.
    # Rapid same-finger movements should always be penalised.
    tempo = note.get('_estimated_tempo', 120.0)
    ioi_threshold = max(0.25, 24.0 / tempo)

    # --- Position shift cost ---
    if pos_diff == 0:
        cost += W['w_position_same']  # bonus (negative)
    else:
        # v13.0: Slide transitions are physically guided, so we treat them as free shifts
        is_slide_shift = (tech in ('slide_up', 'slide_down') and note.get('string') == prev_note.get('string') and finger == prev_finger)
        shift_w = W['w_position_shift_free'] if (is_free_shift or is_slide_shift) else W['w_position_shift']
        
        # v15.0: フレット幅に依存するポジションシフトコストの動的スケーリング (Radicioni 2004 / 物理距離モデル)
        # 高ポジションほどフレット間隔が狭くなり、手の物理的な移動距離が短くなるため、シフトコストを軽減する。
        avg_pos = (pos + prev_pos) / 2.0
        pos_scale = max(0.4, 1.0 - (avg_pos - 1) * 0.05)
        
        if pos < prev_pos:
            # v9.0: Descending shift is biomechanically harder than ascending
            cost += pos_diff * shift_w * W.get('w_descending_shift_factor', 1.3) * pos_scale
        else:
            cost += pos_diff * shift_w * pos_scale

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
        # v13.0: Slide technique bypasses same-finger-different-fret penalty
        is_slide = (tech in ('slide_up', 'slide_down') and note.get('string') == prev_note.get('string'))
        if is_slide:
            cost += W.get('w_tech_slide_bonus', -3.0)
        else:
            # Speed-Adaptive Guide Finger Slide Bonus
            ioi = float(note.get('start', note.get('start_time', 0.0))) - float(prev_note.get('start', prev_note.get('start_time', 0.0)))
            if ioi >= max(0.35, ioi_threshold) and note.get('string') == prev_note.get('string'):
                cost += W['w_guide_finger'] * 0.5
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
    # (Removed to prevent double bonus as it duplicates the check above)

    # --- Pivot finger retention bonus ---
    # 同一の指・同一のフレット・同一 of 弦にとどまる（音の保持・ピボット）場合はボーナス
    if (finger == prev_finger and fret == prev_fret
            and note.get('string') == prev_note.get('string')):
        cost += W.get('w_pivot_finger', 0.0)

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

    # --- v10.0: Left-Right Hand (PIMA-Left Hand) Coordination ---
    r_finger = note.get('r_finger')
    prev_r_finger = prev_note.get('r_finger')
    if r_finger is not None and prev_r_finger is not None:
        # Rule 1: LH Shift / RH Repeat Penalty
        if pos != prev_pos and r_finger == prev_r_finger:
            cost += W.get('w_lh_shift_rh_repeat_penalty', 2.0)

        # Rule 2: LH Pinky / RH Thumb Bass Penalty
        if s >= 5 and finger == 4 and r_finger == 1:
            cost += W.get('w_lh_pinky_rh_thumb_bass_penalty', 3.0)

        # Rule 3: LH Pivot / RH Alternation Bonus
        is_pivot = (finger == prev_finger and fret == prev_fret and s == prev_s)
        if is_pivot and r_finger != prev_r_finger:
            cost += W.get('w_lh_pivot_rh_alternation_bonus', -1.5)

    # --- v13.0: Technique-aware transition cost (Viterbi DP integration) ---
    if tech and tech != 'normal' and s == prev_s:
        # Rule 1: Slide must use same finger on connected notes
        if tech in ('slide_up', 'slide_down'):
            if finger != prev_finger:
                cost += 50.0  # Penalty for using different finger on slide

        # Rule 2: Hammer-on (ascending fret => higher finger)
        elif tech == 'hammer_on':
            if finger > prev_finger and fret > prev_fret:
                # v13.0: Legato involving pinky (4) is harder, so reduce the bonus
                factor = 0.5 if (finger == 4 or prev_finger == 4) else 1.0
                cost += W.get('w_tech_hammer_pull_bonus', -4.0) * factor
            else:
                cost += 50.0  # Penalty for invalid finger order on hammer-on

        # Rule 3: Pull-off (descending fret => lower finger)
        elif tech == 'pull_off':
            if finger < prev_finger and fret < prev_fret:
                factor = 0.5 if (finger == 4 or prev_finger == 4) else 1.0
                cost += W.get('w_tech_hammer_pull_bonus', -4.0) * factor
            else:
                cost += 50.0  # Penalty for invalid finger order on pull-off

    # --- Minimax component (Hori & Sagayama 2016) ---
    # Prevent any single transition from being extremely difficult.
    # Instead of minimizing sum, penalize extreme single-step costs.
    minimax_thresh = W.get('w_minimax_threshold', 50.0)
    if cost > minimax_thresh:
        excess = cost - minimax_thresh
        cost += excess * W.get('w_minimax_excess', 3.0)

    # --- v24.0: Voice leading transition costs ---
    # 1. Common Tone Retention: same pitch class (pc) and same finger/position
    prev_pitch = prev_note.get('pitch', 60)
    curr_pitch = note.get('pitch', 60)
    if prev_pitch % 12 == curr_pitch % 12:
        if finger == prev_finger and pos == prev_pos:
            cost += W.get('w_common_tone_bonus', -4.0)

    # 2. Conjunct Bass Line: smooth transitions on bass strings (5 & 6)
    is_prev_bass = prev_s >= 5
    is_curr_bass = s >= 5
    if is_prev_bass and is_curr_bass:
        pitch_diff = abs(curr_pitch - prev_pitch)
        if 0 < pitch_diff <= 2:
            # Conjunct shift bonus (half/whole step)
            cost += W.get('w_conjunct_bass_bonus', -3.0)
        elif pitch_diff >= 12:
            # Disjunct jump penalty (octave or more)
            cost += W.get('w_disjunct_bass_penalty', 5.0)

    # --- v24.0: Data-driven transition priority (GP5 Prior) ---
    if prev_s == s and prev_fret > 0 and fret > 0:
        run_key = f"{s}-{prev_fret}-{fret}"
        run2_db = _load_mined_patterns()
        if run2_db and run_key in run2_db:
            pat = run2_db[run_key]
            if pat['finger_from'] == prev_finger and pat['finger_to'] == finger:
                cost += math.log(max(2.0, float(pat['count']))) * W.get('w_data_prior', -1.5)

    # --- v14.0: Carlevaro Bass Sustain Rule (音価保持ルール) ---
    prev_s = int(prev_note.get('string', 3))
    if prev_s >= 5:  # 低音弦（5・6弦）
        prev_start = float(prev_note.get('start', prev_note.get('start_time', 0.0)))
        prev_dur = float(prev_note.get('duration', 0.0))
        curr_start = float(note.get('start', note.get('start_time', 0.0)))
        if prev_start + prev_dur > curr_start + 0.05:
            if pos == prev_pos and not (finger == prev_finger and fret != prev_fret):
                cost += W.get('w_bass_sustain_bonus', -3.0)

    return cost


def _viterbi_finger_phrase(fretted_notes: List[dict],
                           free_shift_set: set,
                           next_pos: Optional[int] = None) -> int:
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

    W = _ACTIVE_WEIGHTS

    def _states(note: dict) -> List[Tuple[int, int]]:
        """Return valid (finger, position) states for a note.

        v8.4: Standard 4 states, plus stretch states in high positions
        (fret >= 9) to optimize fingering where frets are narrow.
        """
        fret = int(note.get('fret', 0))
        forced_finger = note.get('_forced_finger')
        if forced_finger:
            pos = fret - (forced_finger - 1)
            return [(forced_finger, max(1, pos))]

        states = []
        seen = set()
        for finger in range(1, 5):
            pos = fret - (finger - 1)
            if pos >= 1:
                states.append((finger, pos))
                seen.add((finger, pos))

        # Allow stretch states in high positions (fret >= 9) where frets are narrow
        if fret >= 9:
            for finger in range(1, 5):
                for offset in [-1, 1]:
                    pos = fret - (finger - 1) + offset
                    if pos >= 1 and (finger, pos) not in seen:
                        # Ensure the hand geometry is realistic
                        actual_offset = fret - pos
                        if 0 <= actual_offset <= 5:
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
        if next_pos is not None and (N - 1) < 3:
            dist = abs(pos - next_pos)
            weight_scale = (3 - (N - 1)) / 3.0
            cost += dist * W.get('w_presentacion_lookahead', 2.0) * weight_scale
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
            if next_pos is not None and (N - 1 - t) < 3:
                dist = abs(pos - next_pos)
                weight_scale = (3 - (N - 1 - t)) / 3.0
                em_cost += dist * W.get('w_presentacion_lookahead', 2.0) * weight_scale
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


def _compute_phrase_features(phrase: List[dict]) -> Tuple[float, float, float]:
    """Compute structural features of a phrase: chord_ratio, arpeggio_ratio, scale_ratio.

    All ratios are in range [0.0, 1.0].
    """
    if len(phrase) <= 1:
        return 0.0, 0.0, 0.0

    simultaneous = 0
    arpeggio_transitions = 0
    scale_transitions = 0

    for i in range(1, len(phrase)):
        prev = phrase[i - 1]
        curr = phrase[i]

        gap = abs(curr.get('start', 0) - prev.get('start', 0))

        # 1. Chord ratio
        if gap <= 0.03:
            simultaneous += 1
            continue  # Simultaneous notes aren't sequential transitions

        # Sequential transition analysis
        prev_s = prev.get('string')
        curr_s = curr.get('string')
        prev_p = prev.get('pitch')
        curr_p = curr.get('pitch')

        if prev_s is None or curr_s is None or prev_p is None or curr_p is None:
            continue

        string_diff = abs(curr_s - prev_s)
        pitch_diff = abs(curr_p - prev_p)

        # 2. Arpeggio transition:
        # Cross strings (string_diff > 0) with relatively small gap (<= 0.25s)
        if string_diff > 0 and gap <= 0.25:
            arpeggio_transitions += 1

        # 3. Scale transition:
        # Small pitch change (<= 2 semitones: whole/half step) with relatively small gap (<= 0.20s)
        elif pitch_diff <= 2 and gap <= 0.20:
            scale_transitions += 1

    N = len(phrase)
    chord_ratio = simultaneous / N

    # Sequential ratio represents what fraction of sequential transitions are of certain type
    seq_steps = N - simultaneous - 1
    if seq_steps > 0:
        arpeggio_ratio = arpeggio_transitions / seq_steps
        scale_ratio = scale_transitions / seq_steps
    else:
        arpeggio_ratio = 0.0
        scale_ratio = 0.0

    return chord_ratio, arpeggio_ratio, scale_ratio


def _blend_weights(chord_ratio: float, arpeggio_ratio: float = 0.0, scale_ratio: float = 0.0) -> dict:
    """Blend solo and chord weights based on chord, arpeggio and scale ratios.

    - High chord_ratio -> Chord weights
    - Low chord_ratio:
      - High arpeggio_ratio -> Boost position same, guide finger, pivot finger
      - High scale_ratio -> Boost finger order, same finger diff penalty
    """
    # Base blending between Solo and Chord weights
    if chord_ratio < 0.5:
        base = _FINGER_DP_WEIGHTS.copy()
    elif chord_ratio > 0.7:
        return _CHORD_DP_WEIGHTS.copy()
    else:
        # Linear blend
        r = (chord_ratio - 0.5) / 0.2
        base = {}
        for key in _FINGER_DP_WEIGHTS:
            solo_val = _FINGER_DP_WEIGHTS[key]
            chord_val = _CHORD_DP_WEIGHTS.get(key, solo_val)
            base[key] = solo_val + r * (chord_val - solo_val)

    # Apply dynamic adjustments based on playing style for solo/hybrid passages
    if chord_ratio < 0.7:
        # 1. Arpeggio adaptation
        if arpeggio_ratio > 0.4:
            arp_factor = min(1.0, (arpeggio_ratio - 0.4) / 0.4)
            # Boost position retention
            base['w_position_same'] += arp_factor * -4.0
            # Boost pivot finger retention
            base['w_pivot_finger'] += arp_factor * -5.0
            # Relax stretch penalty base to allow holding chord shapes
            base['w_stretch_penalty_base'] = max(3.0, base.get('w_stretch_penalty_base', 6.0) - arp_factor * 2.0)

        # 2. Scale Run adaptation
        if scale_ratio > 0.4:
            scale_factor = min(1.0, (scale_ratio - 0.4) / 0.4)
            # Enable and boost sequential finger order (1->2->3->4)
            base['w_finger_order'] = base.get('w_finger_order', 0.0) - scale_factor * 3.0
            # Enable and boost adjacent finger pair smoothness
            base['w_finger_pair_smooth'] = base.get('w_finger_pair_smooth', 0.0) - scale_factor * 2.0
            # Strongly penalise same finger on different fret to encourage alternate fingering
            base['w_same_finger_diff'] += scale_factor * 10.0

    return base


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
        prev_note = current_phrase[-1]
        prev_start = float(prev_note.get('start', 0))
        prev_dur = float(prev_note.get('duration', prev_note.get('end', prev_start) - prev_start))
        if prev_dur <= 0:
            gap = float(note.get('start', 0)) - prev_start
        else:
            gap = float(note.get('start', 0)) - (prev_start + prev_dur)
            
        if gap > phrase_gap:
            phrases.append(current_phrase)
            current_phrase = [note]
        else:
            current_phrase.append(note)
    if current_phrase:
        phrases.append(current_phrase)

    for pi, phrase in enumerate(phrases):
        # v8.3: Context-dependent weight selection
        chord_ratio, arpeggio_ratio, scale_ratio = _compute_phrase_features(phrase)
        _ACTIVE_WEIGHTS = _blend_weights(chord_ratio, arpeggio_ratio, scale_ratio)

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

        # v11.0: Calculate estimated starting position of the next phrase for look-ahead
        next_pos = None
        if pi + 1 < len(phrases):
            next_phrase = phrases[pi + 1]
            next_fretted = [n for n in next_phrase
                            if isinstance(n.get('fret', 0), (int, float))
                            and int(n.get('fret', 0)) > 0]
            if next_fretted:
                next_pos = _estimate_position(next_fretted[:3])

        changed = _viterbi_finger_phrase(fretted_notes, free_shift_set, next_pos=next_pos)
        total_changed += changed

    # Restore default weights
    _ACTIVE_WEIGHTS = _FINGER_DP_WEIGHTS

    return total_changed






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

        # v13.0: Skip if there is an active slide technique on either note
        tech = curr.get('_technique') or curr.get('technique')
        prev_tech = prev.get('_technique') or prev.get('technique')
        if (tech and 'slide' in tech) or (prev_tech and 'slide' in prev_tech):
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


def _enforce_pattern_consistency_lite(notes: list, min_pattern_len: int = 2) -> int:
    """Detect repeated pitch/string/fret patterns and ensure consistent fingering (Lite).
    
    Uses LOWEST CONTEXT COST (正典) across all occurrences with Context Guard:
    1. Group occurrences by (pitch, string, fret) tuple of length >= min_pattern_len
    2. Compute context cost for each occurrence: transition from prev_note + transition to next_note
    3. Choose occurrence with lowest context cost as Canonical (正典)
    4. For other occurrences, apply Canonical fingering ONLY IF:
       a) The target note is not anchored (_is_anchor == False)
       b) The prev position difference is <= 3 frets (Context Guard)
    """
    sorted_notes = sorted(notes, key=lambda n: n.get('start', 0))
    fretted = [n for n in sorted_notes if isinstance(n.get('fret', 0), (int, float)) and int(n.get('fret', 0)) > 0]
    
    if len(fretted) < min_pattern_len * 2:
        return 0
        
    def get_pos(n):
        if not n: return 1
        f = n.get('left_hand_finger', 1)
        if f <= 0: f = 1
        return int(n.get('fret', 0)) - (f - 1)
        
    def calc_transition_cost(n1, n2):
        if not n1 or not n2: return 0.0
        f1, p1 = n1.get('left_hand_finger', 1), get_pos(n1)
        f2, p2 = n2.get('left_hand_finger', 1), get_pos(n2)
        if f1 <= 0: f1 = 1
        if f2 <= 0: f2 = 1
        is_fs = n2.get('start', 0) - n1.get('start', 0) > 0.3
        global _ACTIVE_WEIGHTS
        from finger_assigner import _FINGER_DP_WEIGHTS
        _ACTIVE_WEIGHTS = _FINGER_DP_WEIGHTS
        return _finger_transition_cost_dp(f2, f1, p2, p1, n2, n1, is_fs)

    from collections import defaultdict
    max_len = min(8, len(fretted) // 2)
    changes = 0
    processed_indices = set()
    
    for L in range(max_len, min_pattern_len - 1, -1):
        window_map = defaultdict(list)
        for i in range(len(fretted) - L + 1):
            window = fretted[i:i+L]
            key = tuple((n.get('pitch', 0), n.get('string', 0), n.get('fret', 0)) for n in window)
            window_map[key].append(i)
            
        for key, indices in window_map.items():
            if len(indices) < 2:
                continue
                
            disjoint = []
            last_end = -1
            for idx in indices:
                if idx >= last_end:
                    # check if already processed
                    if not any(idx + j in processed_indices for j in range(L)):
                        disjoint.append(idx)
                        last_end = idx + L
                    
            if len(disjoint) < 2:
                continue
                
            costs = []
            for idx in disjoint:
                prev_n = fretted[idx - 1] if idx > 0 else None
                next_n = fretted[idx + L] if idx + L < len(fretted) else None
                cost_in = calc_transition_cost(prev_n, fretted[idx])
                cost_out = calc_transition_cost(fretted[idx + L - 1], next_n)
                costs.append((cost_in + cost_out, idx))
                
            costs.sort(key=lambda x: x[0])
            canonical_idx = costs[0][1]
            canonical_notes = fretted[canonical_idx:canonical_idx+L]
            canonical_fingers = [n.get('left_hand_finger', 1) for n in canonical_notes]
            
            canonical_prev_n = fretted[canonical_idx - 1] if canonical_idx > 0 else None
            canonical_prev_pos = get_pos(canonical_prev_n) if canonical_prev_n else get_pos(canonical_notes[0])
            
            for idx in disjoint:
                for j in range(L):
                    processed_indices.add(idx + j)
                    
                if idx == canonical_idx:
                    continue
                    
                target_notes = fretted[idx:idx+L]
                target_prev_n = fretted[idx - 1] if idx > 0 else None
                target_prev_pos = get_pos(target_prev_n) if target_prev_n else get_pos(target_notes[0])
                
                if abs(target_prev_pos - canonical_prev_pos) > 3:
                    continue
                    
                for j in range(L):
                    tn = target_notes[j]
                    cf = canonical_fingers[j]
                    if tn.get('left_hand_finger') != cf and not tn.get('_is_anchor', False):
                        tn['left_hand_finger'] = cf
                        changes += 1
                        
    return changes

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


def _mark_bend_support_context(groups: List[List[dict]]):
    """Mark other occupied fingers in the same chord group for bend notes.

    This helps identify whether support fingers (lower indexes than choking finger)
    are occupied by other chord notes.
    """
    for group in groups:
        if len(group) > 1:
            bend_notes = [n for n in group if n.get('_technique') == 'bend' or n.get('technique') == 'bend']
            if not bend_notes:
                continue

            for bn in bend_notes:
                occupied = set()
                for n in group:
                    if n is bn:
                        continue
                    f = n.get('left_hand_finger', 0)
                    if f > 0:
                        occupied.add(f)
                bn['_chord_occupied_fingers'] = occupied


def assign_fingers(notes: List[dict],
                          phrase_gap: float = 0.5,
                          techniques: List[str] = None,
                          detected_key: str = None,
                          use_pattern_consistency_lite: bool = True,
                          use_pitch_proximity: bool = False,
                          use_pivot_fingers: bool = False,
                          forced_fingers: Optional[Dict[Tuple[int, float], int]] = None) -> List[dict]:
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

    # Estimate tempo and assign to all notes for adaptive rules (v9.0)
    tempo = _estimate_tempo(notes)
    for note in notes:
        note['_estimated_tempo'] = tempo

    # Attach technique info to notes if provided
    if techniques and len(techniques) == len(notes):
        for i, note in enumerate(notes):
            note['_technique'] = techniques[i]
    else:
        for note in notes:
            if 'technique' in note and '_technique' not in note:
                note['_technique'] = note['technique']

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

        note_key = (int(note.get("pitch", 0)), round(float(note.get("start", note.get("start_time", 0.0))), 3))
        if forced_fingers and note_key in forced_fingers:
            note['left_hand_finger'] = forced_fingers[note_key]
            note['_forced_finger'] = forced_fingers[note_key]
            note['_is_anchor'] = True
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
    _mark_bend_support_context(groups)
    anchor_count = _mark_anchor_context(notes)
    barre_count = _propagate_barre_context(notes, groups)
    chord_pos_count = _propagate_chord_position(notes, groups)

    # --- v24.0: Scale box matching ---
    try:
        from guitar_fingering_db import match_scale_box
        scale_matches = match_scale_box(notes, detected_key)
        for note_idx, sug_finger, box_name in scale_matches:
            notes[note_idx]['_scale_finger'] = sug_finger
            notes[note_idx]['_scale_box_name'] = box_name
    except Exception as e:
        print(f"  (Warning: Scale box matching failed: {e})")

    # Step 3: Viterbi DP finger assignment (replaces position_smoothing + scale runs)
    viterbi_fixes = _viterbi_finger_dp(notes, phrase_gap=phrase_gap)

    # Step 3.2 (v26.0 - Approach F): Apply corpus-driven phrase templates (human annotations)
    template_fixes = apply_phrase_templates(notes)

    # Note: Post-Viterbi chord re-resolution was tried but caused -12.2%
    # regression by overwriting Viterbi's globally-optimal assignments.
    # Viterbi handles chord notes adequately through context-dependent weights.
    chord_refix = 0

    # Step 3.5: Law 3 — Pitch proximity preserves position
    if use_pitch_proximity:
        prox_fixes = _apply_pitch_proximity_rule(notes)
    else:
        prox_fixes = 0

    # Step 4: Pattern consistency
    if use_pattern_consistency_lite:
        pattern_fixes = _enforce_pattern_consistency_lite(notes)
    else:
        pattern_fixes = 0

    # Step 4.5: Pivot fingers for chord transitions
    if use_pivot_fingers:
        pivot_fixes = _apply_pivot_fingers(notes)
    else:
        pivot_fixes = 0

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
        note.pop('_estimated_tempo', None)
        note.pop('_chord_occupied_fingers', None)
        note.pop('_forced_finger', None)

    # Ensure backward/renderer compatibility: copy left_hand_finger to finger
    for note in notes:
        if 'left_hand_finger' in note:
            note['finger'] = note['left_hand_finger']

    mode = "CNN" if use_cnn else "PDMX"
    print(f"[finger_assigner] {len(notes)} notes ({mode}, "
          f"{len(groups)} groups, viterbi={viterbi_fixes}, templates={template_fixes}, prox={prox_fixes}, "
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

    v13.0: All technique rules (bend, slide, legato) are integrated into
    Viterbi DP emission and transition cost functions to ensure global optimality.
    This post-processing function is deprecated and kept as a placeholder.
    """
    return 0


def _find_prev_fretted(sorted_notes: List[dict], current_idx: int) -> Optional[dict]:
    """Find the previous fretted note (skipping open strings)."""
    for j in range(current_idx - 1, -1, -1):
        if sorted_notes[j].get('fret', 0) > 0:
            return sorted_notes[j]
    return None

