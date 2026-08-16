"""
optuna_finger_weights.py - Optuna Weight Tuning for Finger Assigner Viterbi DP
===============================================================================
Optimizes _FINGER_DP_WEIGHTS in finger_assigner.py using Bayesian search (TPE).

Data sources (priority order):
  1. GP5 files with leftHandFinger annotations (PyGuitarPro)
  2. Synthetic evaluation set from test cases + derived_fingering_rules.json

Usage:
    python optuna_finger_weights.py --trials 200 --output optimized_finger_weights.json

Refs: Sayegh 1989, Miura 2003, Hori & Sagayama 2016, Radicioni & Lombardo 2005
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys
import os
import json
import time
import copy
import argparse
import warnings
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter

# Ensure backend is importable
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GP5_DATA_DIRS = [
    os.path.join(BACKEND_DIR, 'gp5_training', 'data'),
    os.path.join(BACKEND_DIR, 'gp5_training'),
]
DERIVED_RULES_PATH = os.path.join(BACKEND_DIR, 'derived_fingering_rules.json')
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # E4 B3 G3 D3 A2 E2

# Weight keys that are fixed (not tuned)
FIXED_WEIGHTS = {'w_finger_cross'}

# Default solo weights from finger_assigner.py (v9.0/10.0/11.0) — used as baseline
SOLO_DEFAULT_WEIGHTS = {
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
    'w_anchor_penalty': 7.58,
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
}

# Default chord weights from finger_assigner.py — used as baseline for chord target
CHORD_DEFAULT_WEIGHTS = {
    'w_cnn_prior': 10.40,
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
}

DEFAULT_WEIGHTS = SOLO_DEFAULT_WEIGHTS  # Backwards compatibility / placeholder


# Search space bounds for each tunable weight
# Format: (low, high) — Optuna will suggest floats in this range
WEIGHT_BOUNDS = {
    'w_cnn_prior':           (2.0, 45.0),
    'w_offset_rule':         (1.0, 20.0),
    'w_std_offset':          (0.0, 12.0),
    'w_position_same':       (-15.0, 0.0),
    'w_position_shift':      (1.0, 40.0),
    'w_position_shift_free': (0.5, 10.0),
    'w_same_finger_diff':    (1.0, 25.0),
    'w_span_excess':         (1.0, 20.0),
    'w_tendon_coupling':     (0.1, 10.0),
    'w_continuity_2fret':    (-15.0, 0.0),
    'w_guide_finger':        (-25.0, 0.0),
    'w_minimax_threshold':   (20.0, 100.0),
    'w_minimax_excess':      (0.5, 10.0),
    'w_barre_continuity':    (-20.0, 0.0),
    'w_anchor_penalty':      (1.0, 40.0),
    'w_chord_pos_bonus':     (-30.0, 0.0),
    'w_string_cross':        (1.0, 20.0),
    'w_voice_cross_discount':(0.0, 1.0),
    'w_slide_shift_bonus':   (-15.0, 0.0),
    'w_pivot_finger':        (-30.0, 0.0),
    'w_descending_shift_factor': (1.0, 2.5),
    'w_stretch_penalty_base': (2.0, 12.0),
    'w_lh_shift_rh_repeat_penalty': (0.0, 15.0),
    'w_lh_pinky_rh_thumb_bass_penalty': (0.0, 15.0),
    'w_lh_pivot_rh_alternation_bonus': (-10.0, 0.0),
    'w_presentacion_lookahead': (0.0, 10.0),
    'w_tech_slide_bonus': (-15.0, 0.0),
    'w_tech_bend_bonus': (-15.0, 0.0),
    'w_tech_vibrato_pinky_penalty': (0.0, 15.0),
    'w_tech_harmonic_bonus': (-10.0, 0.0),
    'w_tech_hammer_pull_bonus': (-15.0, 0.0),
    'w_bend_support_conflict_penalty': (0.0, 15.0),
    'w_bass_sustain_bonus': (-15.0, 0.0),
    'w_lh_fatigue_penalty': (0.0, 10.0),
    'w_wrist_angle_penalty': (0.0, 15.0),
}

FINGER_NAMES = {0: "Open", 1: "Index", 2: "Middle", 3: "Ring", 4: "Pinky"}


# ============================================================
# Data Loading: GP5 files with leftHandFinger annotations
# ============================================================

def load_gp5_training_data() -> List[Dict[str, Any]]:
    """Load (string, fret, left_hand_finger) tuples from GP5 files.

    Searches GP5_DATA_DIRS for .gp5/.gp4/.gp files and uses PyGuitarPro
    to extract notes with leftHandFinger annotations.

    Returns list of phrase dicts:
      [{'notes': [{'string': s, 'fret': f, 'pitch': p, 'start': t, ...}],
        'ground_truth': [(fret, expected_finger), ...],
        'source': 'gp5:<filename>'}]
    """
    # Try cached JSON first to support environments without raw GP5 files
    cache_path = os.path.join(BACKEND_DIR, 'gp5_training', 'data', 'finger_annotated_notes.json')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            if 'phrases' in cache_data:
                phrases = []
                tuning = STANDARD_TUNING
                for p_idx, cache_p in enumerate(cache_data['phrases']):
                    p_notes = []
                    gt = []
                    source = f"cache_json:{cache_p.get('file', f'phrase_{p_idx}')}"
                    
                    for note in cache_p.get('notes', []):
                        s = note.get('string', 3)
                        fret = note.get('fret', 0)
                        finger = note.get('finger', 0)
                        start = note.get('start', 0.0)
                        
                        pitch = note.get('pitch', 0)
                        if pitch == 0:
                            if 1 <= s <= 6:
                                pitch = tuning[s - 1] + fret
                            else:
                                pitch = 60 + fret
                        
                        p_note = {
                            'string': s,
                            'fret': fret,
                            'pitch': pitch,
                            'start': start,
                        }
                        if 'technique' in note:
                            p_note['technique'] = note['technique']
                        if 'vibrato' in note:
                            p_note['vibrato'] = note['vibrato']
                            
                        p_notes.append(p_note)
                        gt.append((fret, finger))
                        
                    if p_notes:
                        phrases.append({
                            'notes': p_notes,
                            'ground_truth': gt,
                            'source': source
                        })
                
                if phrases:
                    total_notes = sum(len(p['ground_truth']) for p in phrases)
                    print(f"  [CACHE] Loaded {total_notes} annotated notes from {len(phrases)} cached phrases")
                    return phrases
        except Exception as e:
            print(f"  [WARNING] Failed to load cached annotated notes: {e}")

    try:
        import guitarpro
    except ImportError:
        print("  [INFO] PyGuitarPro not installed — skipping GP5 data")
        return []

    gp_files = []
    for data_dir in GP5_DATA_DIRS:
        if not os.path.isdir(data_dir):
            continue
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                ext = fname.lower().rsplit('.', 1)[-1] if '.' in fname else ''
                if ext in ('gp5', 'gp4', 'gp3', 'gp'):
                    gp_files.append(os.path.join(root, fname))

    if not gp_files:
        print("  [INFO] No GP5 files found in training directories")
        return []

    print(f"  Found {len(gp_files)} GP file(s)")
    phrases = []
    total_annotated = 0

    for gp_path in gp_files:
        try:
            song = guitarpro.parse(gp_path)
        except Exception as e:
            continue

        for track in song.tracks:
            # Only standard 6-string guitar
            if len(track.strings) != 6:
                continue

            # Build tuning from track
            tuning = [s.value for s in track.strings]

            for measure in track.measures:
                for voice in measure.voices:
                    for beat in voice.beats:
                        if not beat.notes:
                            continue

                        for note in beat.notes:
                            finger = getattr(note, 'effect', None)
                            if finger is None:
                                continue
                            lhf = getattr(finger, 'leftHandFinger', None)
                            if lhf is None:
                                continue

                            # Map PyGuitarPro finger enum to int
                            finger_int = _parse_lhf(lhf)
                            if finger_int is None or finger_int < 0:
                                continue

                            string_num = note.string  # 1-based
                            fret = note.value

                            # Compute pitch from tuning
                            if 1 <= string_num <= len(tuning):
                                pitch = tuning[string_num - 1] + fret
                            else:
                                pitch = 60 + fret

                            total_annotated += 1

        # For now, collect all annotated notes and group into phrases later
        # (simplified: each file = one phrase batch)

    # Re-parse more carefully, building phrase-level data
    phrases = _extract_phrases_from_gp_files(gp_files)
    if phrases:
        total_notes = sum(len(p['ground_truth']) for p in phrases)
        print(f"  Loaded {total_notes} annotated notes from {len(phrases)} phrases")

    return phrases


def _parse_lhf(lhf) -> Optional[int]:
    """Convert PyGuitarPro LeftHandFinger enum to integer (0-4)."""
    if isinstance(lhf, int):
        return lhf
    # PyGuitarPro uses an enum: open=0, p=1, i=2, m=3, a=4
    # But guitar left hand: 0=open, 1=index, 2=middle, 3=ring, 4=pinky
    name = str(lhf).lower()
    mapping = {'open': 0, 'p': 1, 'i': 1, 'm': 2, 'a': 3, 'c': 4}
    for key, val in mapping.items():
        if key in name:
            return val
    try:
        return int(lhf)
    except (ValueError, TypeError):
        return None


def _extract_phrases_from_gp_files(gp_files: List[str]) -> List[Dict[str, Any]]:
    """Extract annotated phrases from GP files.

    Groups consecutive annotated notes into phrases (gap > 0.5s = new phrase).
    """
    try:
        import guitarpro
    except ImportError:
        return []

    phrases = []

    for gp_path in gp_files:
        try:
            song = guitarpro.parse(gp_path)
        except Exception:
            continue

        for track in song.tracks:
            if len(track.strings) != 6:
                continue

            tuning = [s.value for s in track.strings]
            tempo = 120  # default

            # Collect all annotated notes with timing
            annotated_notes = []
            tick_pos = 0

            for measure in track.measures:
                # Try to get tempo from header
                if hasattr(measure, 'header') and hasattr(measure.header, 'tempo'):
                    tempo = measure.header.tempo.value if hasattr(
                        measure.header.tempo, 'value') else 120

                for voice in measure.voices:
                    beat_tick = tick_pos
                    for beat in voice.beats:
                        duration_ticks = _beat_duration_ticks(beat)
                        time_sec = _ticks_to_seconds(beat_tick, tempo)

                        for note in beat.notes:
                            finger = getattr(note, 'effect', None)
                            lhf = None
                            if finger is not None:
                                lhf = getattr(finger, 'leftHandFinger', None)

                            finger_int = _parse_lhf(lhf) if lhf is not None else None

                            if finger_int is not None and finger_int >= 0:
                                string_num = note.string
                                fret = note.value
                                if 1 <= string_num <= len(tuning):
                                    pitch = tuning[string_num - 1] + fret
                                else:
                                    pitch = 60 + fret

                                annotated_notes.append({
                                    'string': string_num,
                                    'fret': fret,
                                    'pitch': pitch,
                                    'start': time_sec,
                                    'duration': _ticks_to_seconds(duration_ticks, tempo),
                                    'gt_finger': finger_int,
                                })

                        beat_tick += duration_ticks
                tick_pos = beat_tick

            # Group into phrases by time gaps
            if not annotated_notes:
                continue

            annotated_notes.sort(key=lambda n: n['start'])
            current_phrase_notes: List[dict] = [annotated_notes[0]]

            for note in annotated_notes[1:]:
                gap = note['start'] - current_phrase_notes[-1]['start']
                if gap > 0.5:
                    # End current phrase
                    if len(current_phrase_notes) >= 2:
                        phrases.append(_build_phrase_entry(
                            current_phrase_notes, gp_path))
                    current_phrase_notes = [note]
                else:
                    current_phrase_notes.append(note)

            if len(current_phrase_notes) >= 2:
                phrases.append(_build_phrase_entry(
                    current_phrase_notes, gp_path))

    return phrases


def _beat_duration_ticks(beat) -> int:
    """Get beat duration in ticks (960 = quarter note)."""
    try:
        # PyGuitarPro duration values: 1=whole, 2=half, 4=quarter, etc.
        base = 3840 // beat.duration.value if beat.duration.value > 0 else 960
        if beat.duration.isDotted:
            base = int(base * 1.5)
        if beat.duration.tuplet and hasattr(beat.duration.tuplet, 'enters'):
            base = base * beat.duration.tuplet.times // beat.duration.tuplet.enters
        return base
    except Exception:
        return 960


def _ticks_to_seconds(ticks: int, tempo: int) -> float:
    """Convert ticks to seconds (960 ticks per quarter note)."""
    if tempo <= 0:
        tempo = 120
    return ticks / 960.0 * (60.0 / tempo)


def _build_phrase_entry(notes: List[dict], source: str) -> Dict[str, Any]:
    """Build a phrase entry from annotated notes."""
    phrase_notes = []
    ground_truth = []

    for n in notes:
        phrase_notes.append({
            'string': n['string'],
            'fret': n['fret'],
            'pitch': n['pitch'],
            'start': n['start'],
            'duration': n.get('duration', 0.1),
        })
        if n['fret'] > 0:
            ground_truth.append((n['fret'], n['gt_finger']))

    return {
        'notes': phrase_notes,
        'ground_truth': ground_truth,
        'source': f"gp5:{os.path.basename(source)}",
    }


# ============================================================
# Data Loading: Synthetic evaluation set from test cases
# ============================================================

def build_synthetic_eval_set() -> List[Dict[str, Any]]:
    """Build evaluation set from test_finger_assigner.py test cases
    and additional synthetic phrases from derived_fingering_rules.json.

    Each entry is:
      {'notes': [note_dict, ...],
       'ground_truth': [(fret, expected_finger), ...],
       'source': 'test:TestName' or 'synthetic:...'}
    """
    eval_set: List[Dict[str, Any]] = []

    # --- 1. Hardcoded test cases (from test_finger_assigner.py) ---
    test_cases = _get_test_cases()
    eval_set.extend(test_cases)

    # --- 2. Synthetic phrases from derived rules + position knowledge ---
    synthetic = _generate_synthetic_phrases()
    eval_set.extend(synthetic)

    return eval_set


def _get_test_cases() -> List[Dict[str, Any]]:
    """Extract test cases with expected finger assignments."""
    cases = []

    # Test 3: C major chord
    cases.append({
        'notes': [
            {"string": 5, "fret": 3, "pitch": 48, "start": 0.0},
            {"string": 4, "fret": 2, "pitch": 52, "start": 0.0},
            {"string": 3, "fret": 0, "pitch": 55, "start": 0.0},
            {"string": 2, "fret": 1, "pitch": 60, "start": 0.0},
            {"string": 1, "fret": 0, "pitch": 64, "start": 0.0},
        ],
        'ground_truth': [(3, 3), (2, 2), (1, 1)],
        'source': 'test:C_major_chord',
    })

    # Test 5: Position shift 1-2-3 → 5-7-8
    cases.append({
        'notes': [
            {"string": 1, "fret": 1, "pitch": 65, "start": 0.0},
            {"string": 1, "fret": 2, "pitch": 66, "start": 0.15},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.30},
            {"string": 1, "fret": 5, "pitch": 69, "start": 1.2},
            {"string": 1, "fret": 7, "pitch": 71, "start": 1.35},
            {"string": 1, "fret": 8, "pitch": 72, "start": 1.50},
        ],
        'ground_truth': [(1, 1), (2, 2), (3, 3), (5, 1), (7, 3), (8, 4)],
        'source': 'test:position_shift_1_2_3_to_5_7_8',
    })

    # Test 6: Position consistency (pos=5)
    cases.append({
        'notes': [
            {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
            {"string": 2, "fret": 6, "pitch": 65, "start": 0.15},
            {"string": 1, "fret": 7, "pitch": 71, "start": 0.30},
            {"string": 2, "fret": 8, "pitch": 67, "start": 0.45},
            {"string": 1, "fret": 5, "pitch": 69, "start": 0.60},
        ],
        'ground_truth': [(5, 1), (6, 2), (7, 3), (8, 4), (5, 1)],
        'source': 'test:position_consistency_pos5',
    })

    # Test 9: First position basic 1-2-3-4
    cases.append({
        'notes': [
            {"string": 1, "fret": 1, "pitch": 65, "start": 0.0},
            {"string": 1, "fret": 2, "pitch": 66, "start": 0.15},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.30},
            {"string": 1, "fret": 4, "pitch": 68, "start": 0.45},
        ],
        'ground_truth': [(1, 1), (2, 2), (3, 3), (4, 4)],
        'source': 'test:first_position_1_2_3_4',
    })

    # Additional: Descending scale 8-7-5-3 (pos=5: 8→4, 7→3, 5→1)
    # then shift to pos=3: 3→1
    cases.append({
        'notes': [
            {"string": 1, "fret": 8, "pitch": 72, "start": 0.0},
            {"string": 1, "fret": 7, "pitch": 71, "start": 0.15},
            {"string": 1, "fret": 5, "pitch": 69, "start": 0.30},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.45},
        ],
        'ground_truth': [(8, 4), (7, 3), (5, 1), (3, 1)],
        'source': 'test:descending_8_7_5_3',
    })

    # Additional: G major chord (position 1)
    cases.append({
        'notes': [
            {"string": 6, "fret": 3, "pitch": 43, "start": 0.0},
            {"string": 5, "fret": 2, "pitch": 47, "start": 0.0},
            {"string": 4, "fret": 0, "pitch": 50, "start": 0.0},
            {"string": 3, "fret": 0, "pitch": 55, "start": 0.0},
            {"string": 2, "fret": 0, "pitch": 59, "start": 0.0},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.0},
        ],
        'ground_truth': [(3, 3), (2, 2), (3, 4)],
        'source': 'test:G_major_chord',
    })

    # Additional: Am chord (position 1)
    cases.append({
        'notes': [
            {"string": 5, "fret": 0, "pitch": 45, "start": 0.0},
            {"string": 4, "fret": 2, "pitch": 52, "start": 0.0},
            {"string": 3, "fret": 2, "pitch": 57, "start": 0.0},
            {"string": 2, "fret": 1, "pitch": 60, "start": 0.0},
            {"string": 1, "fret": 0, "pitch": 64, "start": 0.0},
        ],
        'ground_truth': [(2, 2), (2, 3), (1, 1)],
        'source': 'test:Am_chord',
    })

    # Additional: Position 7, scale run
    cases.append({
        'notes': [
            {"string": 2, "fret": 7, "pitch": 66, "start": 0.0},
            {"string": 2, "fret": 8, "pitch": 67, "start": 0.15},
            {"string": 2, "fret": 10, "pitch": 69, "start": 0.30},
            {"string": 1, "fret": 7, "pitch": 71, "start": 0.45},
            {"string": 1, "fret": 8, "pitch": 72, "start": 0.60},
            {"string": 1, "fret": 10, "pitch": 74, "start": 0.75},
        ],
        'ground_truth': [(7, 1), (8, 2), (10, 4), (7, 1), (8, 2), (10, 4)],
        'source': 'test:scale_run_pos7',
    })

    # Additional: Barre F chord
    cases.append({
        'notes': [
            {"string": 6, "fret": 1, "pitch": 41, "start": 0.0},
            {"string": 5, "fret": 3, "pitch": 48, "start": 0.0},
            {"string": 4, "fret": 3, "pitch": 53, "start": 0.0},
            {"string": 3, "fret": 2, "pitch": 57, "start": 0.0},
            {"string": 2, "fret": 1, "pitch": 60, "start": 0.0},
            {"string": 1, "fret": 1, "pitch": 65, "start": 0.0},
        ],
        'ground_truth': [(1, 1), (3, 3), (3, 4), (2, 2), (1, 1), (1, 1)],
        'source': 'test:F_barre_chord',
    })

    # Additional: Wide position shift (pos 1 → pos 9)
    cases.append({
        'notes': [
            {"string": 1, "fret": 1, "pitch": 65, "start": 0.0},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.15},
            {"string": 1, "fret": 9, "pitch": 73, "start": 1.0},
            {"string": 1, "fret": 10, "pitch": 74, "start": 1.15},
            {"string": 1, "fret": 12, "pitch": 76, "start": 1.30},
        ],
        'ground_truth': [(1, 1), (3, 3), (9, 1), (10, 2), (12, 4)],
        'source': 'test:wide_position_shift',
    })

    return cases


def _generate_synthetic_phrases() -> List[Dict[str, Any]]:
    """Generate synthetic evaluation phrases from derived_fingering_rules.json.

    Uses the fret_offset_rules (mined from 3,283 chord voicings) to create
    position-based ground truth for various fret patterns.
    """
    phrases = []

    # Load derived rules for validation
    rules_data = _load_derived_rules_data()
    if not rules_data:
        return phrases

    # Generate phrases for each position from 1 to 12
    for pos in range(1, 13):
        # Ascending chromatic within position
        notes = []
        gt = []
        for finger in range(1, 5):  # 1-4
            fret = pos + (finger - 1)
            if fret > 22:
                continue
            string = 1 if finger <= 2 else 2
            pitch = STANDARD_TUNING[string - 1] + fret
            notes.append({
                'string': string,
                'fret': fret,
                'pitch': pitch,
                'start': (finger - 1) * 0.15,
            })
            gt.append((fret, finger))

        if len(gt) >= 3:
            phrases.append({
                'notes': notes,
                'ground_truth': gt,
                'source': f'synthetic:chromatic_pos{pos}',
            })

    # Generate cross-string scale patterns
    for pos in [1, 3, 5, 7, 9]:
        notes = []
        gt = []
        t = 0.0
        for s in [3, 2, 1]:  # Ascending across strings
            for offset in [0, 1, 2, 3]:
                fret = pos + offset
                finger = offset + 1
                pitch = STANDARD_TUNING[s - 1] + fret
                notes.append({
                    'string': s, 'fret': fret,
                    'pitch': pitch, 'start': t,
                })
                gt.append((fret, finger))
                t += 0.12
        if len(gt) >= 6:
            phrases.append({
                'notes': notes,
                'ground_truth': gt,
                'source': f'synthetic:cross_string_pos{pos}',
            })

    # Generate common chord-to-melody patterns
    # E.g.: C chord → melody on string 1
    phrases.append({
        'notes': [
            {"string": 5, "fret": 3, "pitch": 48, "start": 0.0},
            {"string": 4, "fret": 2, "pitch": 52, "start": 0.0},
            {"string": 2, "fret": 1, "pitch": 60, "start": 0.0},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.5},
            {"string": 1, "fret": 1, "pitch": 65, "start": 0.65},
        ],
        'ground_truth': [(3, 3), (2, 2), (1, 1), (3, 3), (1, 1)],
        'source': 'synthetic:chord_to_melody_C',
    })

    # Hammer-on / pull-off patterns
    for base_fret in [1, 5, 7]:
        t = 0.0
        notes = []
        gt = []
        for fret, finger in [(base_fret, 1), (base_fret + 2, 3),
                             (base_fret, 1), (base_fret + 3, 4)]:
            pitch = STANDARD_TUNING[0] + fret
            notes.append({
                'string': 1, 'fret': fret,
                'pitch': pitch, 'start': t,
            })
            gt.append((fret, finger))
            t += 0.1
        phrases.append({
            'notes': notes,
            'ground_truth': gt,
            'source': f'synthetic:hammer_pull_pos{base_fret}',
        })

    # --- v17.0: Synthetic phrases for newly introduced rules (v13.0 - v16.0) ---

    # 1. Bass sustain prolongation
    phrases.append({
        'notes': [
            {"string": 6, "fret": 3, "pitch": 43, "start": 0.0, "duration": 1.0},
            {"string": 2, "fret": 1, "pitch": 60, "start": 0.2, "duration": 0.2},
            {"string": 1, "fret": 3, "pitch": 67, "start": 0.4, "duration": 0.2},
            {"string": 1, "fret": 1, "pitch": 65, "start": 0.6, "duration": 0.2},
        ],
        'ground_truth': [(3, 3), (1, 1), (3, 3), (1, 1)],
        'source': 'synthetic:bass_sustain_prolongation',
    })

    # 2. High position shift easing
    phrases.append({
        'notes': [
            {"string": 1, "fret": 12, "pitch": 76, "start": 0.0},
            {"string": 1, "fret": 14, "pitch": 78, "start": 0.15},
            {"string": 1, "fret": 15, "pitch": 79, "start": 0.30},
            {"string": 1, "fret": 17, "pitch": 81, "start": 0.45},
        ],
        'ground_truth': [(12, 1), (14, 3), (15, 4), (17, 3)],
        'source': 'synthetic:high_position_shift_easing',
    })

    # 3. Fatigue and wrist twist avoidance
    phrases.append({
        'notes': [
            {"string": 5, "fret": 10, "pitch": 50, "start": 0.0},
            {"string": 5, "fret": 12, "pitch": 52, "start": 0.15},
            {"string": 5, "fret": 8, "pitch": 48, "start": 0.30},
        ],
        'ground_truth': [(10, 3), (12, 4), (8, 1)],
        'source': 'synthetic:fatigue_wrist_avoidance',
    })

    # 4. Slide same finger
    phrases.append({
        'notes': [
            {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
            {"string": 1, "fret": 7, "pitch": 71, "start": 0.1, "technique": "slide_up"},
        ],
        'ground_truth': [(5, 1), (7, 1)],
        'source': 'synthetic:slide_same_finger',
    })

    # 5. Legato hammer-on
    phrases.append({
        'notes': [
            {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
            {"string": 1, "fret": 7, "pitch": 71, "start": 0.15, "technique": "hammer_on"},
        ],
        'ground_truth': [(5, 1), (7, 3)],
        'source': 'synthetic:legato_hammer_on',
    })

    return phrases


def _load_derived_rules_data() -> Optional[dict]:
    """Load derived_fingering_rules.json if available."""
    if not os.path.exists(DERIVED_RULES_PATH):
        return None
    try:
        with open(DERIVED_RULES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# Evaluation Engine
# ============================================================

def evaluate_weights(weights_dict: Dict[str, float],
                     eval_set: List[Dict[str, Any]],
                     target: str = 'solo',
                     verbose: bool = False) -> Tuple[float, float]:
    """Evaluate a set of weights against the evaluation set.

    Temporarily overrides finger_assigner._FINGER_DP_WEIGHTS or _CHORD_DP_WEIGHTS,
    runs assign_fingers() on each phrase, and compares with ground truth.

    Returns:
        (accuracy, position_consistency_score)
        accuracy: fraction of notes with correct finger assignment
        position_consistency: 1.0 - normalized unique-positions-per-phrase
    """
    import finger_assigner as fa

    # Save original weights and suppress print output
    original_solo_weights = dict(fa._FINGER_DP_WEIGHTS)
    original_chord_weights = dict(fa._CHORD_DP_WEIGHTS)
    original_print = fa.__builtins__ if hasattr(fa, '__builtins__') else None

    # Override weights
    if target == 'solo':
        for k, v in weights_dict.items():
            if k in fa._FINGER_DP_WEIGHTS:
                fa._FINGER_DP_WEIGHTS[k] = v
    else:
        for k, v in weights_dict.items():
            if k in fa._CHORD_DP_WEIGHTS:
                fa._CHORD_DP_WEIGHTS[k] = v

    # Suppress finger_assigner's print statements during evaluation
    import builtins
    _real_print = builtins.print

    def _silent_print(*args, **kwargs):
        pass

    builtins.print = _silent_print

    try:
        total_correct = 0
        total_notes = 0
        position_scores = []

        for entry in eval_set:
            notes = copy.deepcopy(entry['notes'])
            gt = entry['ground_truth']

            try:
                result = fa.assign_fingers(notes)
            except Exception as e:
                if verbose:
                    _real_print(f"  Error on {entry['source']}: {e}")
                continue

            # Compare fretted notes with ground truth
            fretted = [n for n in result
                       if isinstance(n.get('fret', 0), (int, float))
                       and int(n.get('fret', 0)) > 0]

            for i, (exp_fret, exp_finger) in enumerate(gt):
                if i >= len(fretted):
                    break
                actual_fret = int(fretted[i].get('fret', 0))
                actual_finger = int(fretted[i].get('left_hand_finger', -1))

                # Only count if fret matches (same note)
                if actual_fret == exp_fret:
                    total_notes += 1
                    if actual_finger == exp_finger:
                        total_correct += 1

            # Position consistency: count unique positions in phrase
            positions = set()
            for n in fretted:
                fret = int(n.get('fret', 0))
                finger = int(n.get('left_hand_finger', 1))
                if fret > 0 and finger > 0:
                    pos = fret - (finger - 1)
                    if pos >= 1:
                        positions.add(pos)

            if len(fretted) > 0:
                # Normalize: 1 position = 1.0 score, many positions = lower
                n_unique = len(positions)
                max_positions = max(len(fretted), 1)
                consistency = 1.0 - (n_unique - 1) / max(max_positions, 2)
                position_scores.append(max(0.0, consistency))

        accuracy = total_correct / total_notes if total_notes > 0 else 0.0
        avg_consistency = (sum(position_scores) / len(position_scores)
                          if position_scores else 0.0)

        if verbose:
            _real_print(f"  Accuracy: {accuracy*100:.1f}% "
                        f"({total_correct}/{total_notes}), "
                        f"Consistency: {avg_consistency:.3f}")

    finally:
        # Restore original weights and print
        for k, v in original_solo_weights.items():
            fa._FINGER_DP_WEIGHTS[k] = v
        for k, v in original_chord_weights.items():
            fa._CHORD_DP_WEIGHTS[k] = v
        builtins.print = _real_print

    return accuracy, avg_consistency


# ============================================================
# Optuna Objective
# ============================================================

def create_objective(eval_set: List[Dict[str, Any]],
                     target: str = 'solo',
                     consistency_weight: float = 0.1,
                     default_weights: Dict[str, float] = None):
    """Create an Optuna objective function.

    Maximizes: accuracy + consistency_weight * position_consistency
    with strict regression constraint on synthetic/test data.
    """
    if default_weights is None:
        default_weights = SOLO_DEFAULT_WEIGHTS if target == 'solo' else CHORD_DEFAULT_WEIGHTS

    # Split dataset: synthetic/test (regression cases) vs real cache
    synthetic_eval = [p for p in eval_set if 'synthetic' in p['source'] or 'test' in p['source']]
    real_eval = [p for p in eval_set if 'synthetic' not in p['source'] and 'test' not in p['source']]

    best_acc = [0.0]
    best_weights = [{}]
    trial_count = [0]

    def objective(trial):
        trial_count[0] += 1

        # Build weights dict from trial suggestions
        w = {}
        for key, (lo, hi) in WEIGHT_BOUNDS.items():
            w[key] = trial.suggest_float(key, lo, hi)

        # Fixed weights
        for key in FIXED_WEIGHTS:
            w[key] = default_weights.get(key, 200.0)

        # 1. Evaluate synthetic/test regression cases
        syn_acc, syn_cons = evaluate_weights(w, synthetic_eval, target=target)
        
        # 2. Evaluate real cache data if available
        real_acc, real_cons = evaluate_weights(w, real_eval, target=target) if real_eval else (syn_acc, syn_cons)

        # Soft weighted average score: 30% synthetic (regression prevention) / 70% real cache
        accuracy = syn_acc * 0.3 + real_acc * 0.7
        consistency = syn_cons * 0.3 + real_cons * 0.7

        score = accuracy + consistency_weight * consistency

        # Track best based on weighted accuracy
        if accuracy > best_acc[0]:
            best_acc[0] = accuracy
            best_weights[0] = dict(w)
            print(f"  ★ New best: {accuracy*100:.1f}% "
                  f"(consistency={consistency:.3f}, trial {trial.number})")

        # Progress reporting every 20 trials
        if trial_count[0] % 20 == 0:
            print(f"  [Progress] Trial {trial_count[0]}: "
                  f"best_acc={best_acc[0]*100:.1f}%, "
                  f"current_real={real_acc*100:.1f}%, syn_acc={syn_acc*100:.1f}%")

        return score

    return objective, best_acc, best_weights


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optuna weight tuning for finger_assigner Viterbi DP')
    parser.add_argument('--target', type=str, choices=['solo', 'chord'], default='solo',
                        help="Optimization target: 'solo' or 'chord' (default: 'solo')")
    parser.add_argument('--trials', type=int, default=200,
                        help='Number of Optuna trials (default: 200)')
    parser.add_argument('--output', type=str,
                        default='optimized_finger_weights.json',
                        help='Output JSON file for best weights')
    parser.add_argument('--consistency-weight', type=float, default=0.1,
                        help='Weight for position consistency in objective '
                             '(default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for TPESampler (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed evaluation info')
    args = parser.parse_args()

    # Determine default weights and output filename based on target
    target_weights = CHORD_DEFAULT_WEIGHTS if args.target == 'chord' else SOLO_DEFAULT_WEIGHTS
    if args.output == 'optimized_finger_weights.json' and args.target == 'chord':
        args.output = 'optimized_chord_weights.json'

    print("=" * 60)
    print("  Optuna Finger Weight Optimizer")
    print(f"  Target: {args.target.upper()}")
    print("  (Viterbi DP cost function weights)")
    print("=" * 60)

    # --- Step 1: Load training data ---
    print("\n[1/4] Loading evaluation data...")

    eval_set: List[Dict[str, Any]] = []

    # Try GP5 files first
    gp5_data = load_gp5_training_data()
    if gp5_data:
        eval_set.extend(gp5_data)
        print(f"  GP5 data: {len(gp5_data)} phrases, "
              f"{sum(len(p['ground_truth']) for p in gp5_data)} notes")

    # Always add synthetic/test data (for coverage)
    synthetic_data = build_synthetic_eval_set()
    eval_set.extend(synthetic_data)
    print(f"  Synthetic/test data: {len(synthetic_data)} phrases, "
          f"{sum(len(p['ground_truth']) for p in synthetic_data)} notes")

    total_phrases = len(eval_set)
    total_gt_notes = sum(len(p['ground_truth']) for p in eval_set)
    print(f"  Total: {total_phrases} phrases, {total_gt_notes} ground truth notes")

    if total_gt_notes == 0:
        print("ERROR: No evaluation data available!")
        sys.exit(1)

    # Data source summary
    sources = Counter(p['source'].split(':')[0] for p in eval_set)
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt} phrases")

    # --- Step 2: Baseline evaluation ---
    print("\n[2/4] Evaluating baseline weights...")
    baseline_acc, baseline_cons = evaluate_weights(
        target_weights, eval_set, target=args.target, verbose=True)
    print(f"  Baseline accuracy:    {baseline_acc*100:.1f}%")
    print(f"  Baseline consistency: {baseline_cons:.3f}")

    # --- Step 3: Optuna optimization ---
    if args.trials <= 0:
        print("\nSkipping optimization (--trials 0). Baseline results above.")
        return

    print(f"\n[3/4] Running Optuna optimization ({args.trials} trials)...")

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective_fn, best_acc_ref, best_weights_ref = create_objective(
        eval_set, target=args.target, consistency_weight=args.consistency_weight, default_weights=target_weights)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Enqueue the default weights as the first trial (warm start)
    default_params = {}
    for key, (lo, hi) in WEIGHT_BOUNDS.items():
        val = target_weights.get(key, (lo + hi) / 2)
        # Clamp to bounds
        val = max(lo, min(hi, val))
        default_params[key] = val
    study.enqueue_trial(default_params)

    t0 = time.time()
    study.optimize(objective_fn, n_trials=args.trials, show_progress_bar=False)
    t1 = time.time()

    # --- Step 4: Results ---
    print(f"\n{'=' * 60}")
    print(f"  OPTIMIZATION COMPLETE ({t1 - t0:.0f}s)")
    print(f"{'=' * 60}")

    best_trial = study.best_trial
    best_acc = best_acc_ref[0]
    best_weights = best_weights_ref[0]

    # If best_weights is empty (shouldn't happen), use study's best params
    if not best_weights:
        best_weights = dict(best_trial.params)
        for key in FIXED_WEIGHTS:
            best_weights[key] = target_weights.get(key, 200.0)

    # Final evaluation with best weights
    final_acc, final_cons = evaluate_weights(
        best_weights, eval_set, target=args.target, verbose=True)

    print(f"\n  Baseline accuracy:    {baseline_acc * 100:.1f}%")
    print(f"  Best accuracy:        {final_acc * 100:.1f}%")
    print(f"  Improvement:          {(final_acc - baseline_acc) * 100:+.1f}%")
    print(f"  Position consistency: {final_cons:.3f}")
    print(f"  Total trials:         {args.trials}")
    print(f"  Time elapsed:         {t1 - t0:.0f}s")

    # Print weight comparison
    print(f"\n  Weight comparison (default -> optimized):")
    for key in sorted(best_weights.keys()):
        default_val = target_weights.get(key, '?')
        new_val = best_weights[key]
        if isinstance(default_val, (int, float)):
            changed = " *" if abs(new_val - default_val) > 0.5 else ""
            print(f"    {key:<30} {default_val:>8.2f} -> {new_val:>8.2f}{changed}")
        else:
            print(f"    {key:<30} {'?':>8} -> {new_val:>8.2f}")

    # Save results
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(BACKEND_DIR, output_path)

    output = {
        'weights': best_weights,
        'finger_accuracy': final_acc,
        'position_consistency': final_cons,
        'baseline_accuracy': baseline_acc,
        'baseline_consistency': baseline_cons,
        'n_trials': args.trials,
        'n_eval_phrases': total_phrases,
        'n_eval_notes': total_gt_notes,
        'data_sources': dict(sources),
        'method': 'finger_viterbi_dp_optuna',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fixed_weights': {k: target_weights.get(k, 200.0) for k in FIXED_WEIGHTS},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")

    # --- Integration instructions ---
    print(f"\n{'=' * 60}")
    print(f"  Integration Instructions")
    print(f"{'=' * 60}")
    
    target_var = '_CHORD_DP_WEIGHTS' if args.target == 'chord' else '_FINGER_DP_WEIGHTS'
    opt_file = 'optimized_chord_weights.json' if args.target == 'chord' else 'optimized_finger_weights.json'
    
    print(f"""
  To apply the optimized weights to finger_assigner.py:

  Option A: Direct replacement
    1. Open finger_assigner.py
    2. Replace the {target_var} dictionary
       with the following:

    {target_var} = {{""")
    for key in sorted(best_weights.keys()):
        val = best_weights[key]
        print(f"        '{key}': {val:.4f},")
    print(f"""    }}

  Option B: Auto-load from JSON
    Add this to finger_assigner.py after {target_var} definition:

    _OPT_PATH = os.path.join(os.path.dirname(__file__),
                              '{opt_file}')
    if os.path.exists(_OPT_PATH):
        with open(_OPT_PATH, 'r') as _f:
            _opt = json.load(_f)
        if 'weights' in _opt:
            {target_var}.update(_opt['weights'])
            print(f"[finger_assigner] Loaded optimized weights from {{_OPT_PATH}}")
""")


if __name__ == '__main__':
    main()
