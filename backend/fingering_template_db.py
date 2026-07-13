import os
import json
from typing import List, Tuple, Dict, Any

_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_DIR, 'gp5_training', 'data', 'fingering_templates.json')

_TEMPLATES = None  # Loaded dict: L -> key -> {fingers, count}
TUNING = [64, 59, 55, 50, 45, 40]

def _get_pitch(note):
    p = note.get('pitch', 0)
    if p > 0:
        return p
    s = note.get('string', 3)
    f = note.get('fret', 0)
    if 1 <= s <= 6:
        return TUNING[s-1] + f
    return 60

def load_db():
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
        
    # Check absolute path fallback (due to import contexts)
    db_path = _DB_PATH
    if not os.path.exists(db_path):
        # Fallback to parent directory layout
        parent_dir = os.path.dirname(_DIR)
        fallback = os.path.join(parent_dir, 'backend', 'gp5_training', 'data', 'fingering_templates.json')
        if os.path.exists(fallback):
            db_path = fallback
        else:
            # Fallback 2: Check current directory relative
            rel_path = os.path.join('gp5_training', 'data', 'fingering_templates.json')
            if os.path.exists(rel_path):
                db_path = rel_path
            else:
                print(f"[template_db] Warning: fingering_templates.json not found anywhere.")
                _TEMPLATES = {4: {}, 5: {}, 6: {}}
                return _TEMPLATES
                
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            # Convert keys to int
            _TEMPLATES = {int(k): v for k, v in raw.items()}
            print(f"[template_db] Loaded templates: L=4 ({len(_TEMPLATES.get(4, {}))}), L=5 ({len(_TEMPLATES.get(5, {}))}), L=6 ({len(_TEMPLATES.get(6, {}))})")
    except Exception as e:
        print(f"[template_db] load failed: {e}")
        _TEMPLATES = {4: {}, 5: {}, 6: {}}
    return _TEMPLATES

def apply_phrase_templates(notes: List[dict]) -> int:
    """Scan notes for matching templates and apply the fingers directly.

    Modifies notes list in-place.
    Returns the number of notes overwritten.
    """
    templates = load_db()
    N = len(notes)
    if N < 4:
        return 0

    pitches = [_get_pitch(n) for n in notes]
    strings = [n.get('string', 3) for n in notes]
    frets = [n.get('fret', 0) for n in notes]

    # Format: index -> (finger, confidence_count, match_length)
    overwrites: Dict[int, Tuple[int, int, int]] = {}

    for L in [6, 5, 4]:
        if L > N:
            continue
            
        for i in range(N - L + 1):
            win_notes = notes[i:i+L]
            if not all(win_notes[j].get('fret', 0) > 0 for j in range(L)):
                continue
                
            win_pitches = pitches[i:i+L]
            win_strings = strings[i:i+L]
            win_frets = frets[i:i+L]
            
            p_diff = [win_pitches[j] - win_pitches[j-1] for j in range(1, L)]
            s_diff = [win_strings[j] - win_strings[j-1] for j in range(1, L)]
            f_diff = [win_frets[j] - win_frets[j-1] for j in range(1, L)]
            
            key = f"pitch:{p_diff}|string:{s_diff}|fret:{f_diff}"
            
            db_entry = templates.get(L, {}).get(key)
            if db_entry:
                suggested_fingers = db_entry['fingers']
                count = db_entry['count']
                
                can_apply = True
                for offset in range(L):
                    idx = i + offset
                    if idx in overwrites:
                        existing_fingers, existing_count, existing_len = overwrites[idx]
                        if L < existing_len:
                            can_apply = False
                            break
                        elif L == existing_len and count <= existing_count:
                            can_apply = False
                            break
                            
                if can_apply:
                    for offset in range(L):
                        idx = i + offset
                        overwrites[idx] = (suggested_fingers[offset], count, L)

    changes = 0
    for idx, (finger, count, L) in overwrites.items():
        note = notes[idx]
        old_finger = note.get('left_hand_finger', 0)
        if old_finger != finger:
            note['left_hand_finger'] = finger
            changes += 1
            
    return changes
