import math
from typing import List, Tuple, Dict, Any
from chord_theory import _parse_chord_name, _get_chord_notes_pc

# Pitch classes (0=C, 1=C#, ..., 11=B)
KEY_SIGNATURES = {
    "C":  [0, 2, 4, 5, 7, 9, 11],
    "G":  [0, 2, 4, 6, 7, 9, 11],
    "D":  [1, 2, 4, 6, 7, 9, 11],
    "A":  [1, 2, 4, 6, 8, 9, 11],
    "E":  [1, 3, 4, 6, 8, 9, 11],
    "B":  [1, 3, 4, 6, 8, 10, 11],
    "F":  [0, 2, 4, 5, 7, 9, 10],
    "Bb": [0, 2, 3, 5, 7, 9, 10],
    "Eb": [0, 2, 3, 5, 7, 8, 10],
    "Ab": [0, 1, 3, 5, 7, 8, 10],
    "Am": [0, 2, 4, 5, 7, 9, 11],
    "Em": [0, 2, 4, 6, 7, 9, 11],
    "Dm": [0, 2, 3, 5, 7, 9, 10],
    "Bm": [1, 2, 4, 6, 7, 9, 11],
}

def _group_simultaneous_notes(notes: List[dict], time_tolerance: float = 0.05) -> List[List[dict]]:
    """Group notes that start at roughly the same time (chords)."""
    if not notes:
        return []
    
    sorted_notes = sorted(notes, key=lambda n: n["start"])
    groups = []
    current_group = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        if note["start"] - current_group[0]["start"] <= time_tolerance:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)
    return groups

def _get_chord_at_time(chords: List[dict], t_sec: float) -> str:
    if not chords:
        return "N.C."
    for c in chords:
        start = c.get('start', 0.0)
        end = c.get('end', 9999.0)
        if start <= t_sec < end:
            return c.get('chord', "N.C.")
    return "N.C."

def _pass1_melodic_smoothing(notes: List[dict], logs: List[Dict[str, Any]]) -> List[dict]:
    """
    Pass 1: Octave error correction (Melodic Smoothing)
    Only applies to single-note melodic lines.
    Excludes base notes (E3 / MIDI 52 and below).
    """
    if not notes:
        return notes
        
    sorted_notes = sorted(notes, key=lambda n: n["start"])
    groups = _group_simultaneous_notes(sorted_notes, time_tolerance=0.08)
    
    for i in range(1, len(groups) - 1):
        prev_g = groups[i-1]
        curr_g = groups[i]
        next_g = groups[i+1]
        
        # Only apply to single-note melodies (not chords)
        if len(prev_g) == 1 and len(curr_g) == 1 and len(next_g) == 1:
            prev_n = prev_g[0]
            curr_n = curr_g[0]
            next_n = next_g[0]
            
            p1 = prev_n["pitch"]
            p2 = curr_n["pitch"]
            p3 = next_n["pitch"]
            
            # Exclude base notes and notes following base notes
            if p1 < 52 or p2 < 52 or p3 < 52:
                continue
            
            jump1 = p2 - p1
            jump2 = p3 - p2
            
            # Check for octave jumps (+/- 12 semitones roughly)
            if 10 <= abs(jump1) <= 14:
                # If we shift p2 by -octave, does it smooth the line?
                sign = -1 if jump1 > 0 else 1
                shifted_p2 = p2 + (sign * 12)
                
                # Check smoothness (stepwise or small leaps < 5 semitones)
                smooth_jump1 = abs(shifted_p2 - p1)
                smooth_jump2 = abs(p3 - shifted_p2)
                
                original_smoothness = abs(jump1) + abs(jump2)
                new_smoothness = smooth_jump1 + smooth_jump2
                
                # If shifting heavily improves smoothness and connects well
                if new_smoothness < original_smoothness and smooth_jump1 <= 5 and smooth_jump2 <= 5:
                    logs.append({
                        "pass": 1,
                        "action": "modify",
                        "note": dict(curr_n),
                        "reason": f"Melodic smoothing (p1={p1}, p2={p2}->{shifted_p2}, p3={p3})",
                        "modified_pitch": shifted_p2
                    })
                    curr_n["pitch"] = shifted_p2

    return sorted_notes

def _pass2_harmonic_merging(notes: List[dict], logs: List[Dict[str, Any]]) -> List[dict]:
    """
    Pass 2: Harmonic Merging (Resonance / Overtone deduplication)
    """
    groups = _group_simultaneous_notes(notes, time_tolerance=0.05)
    kept_notes = []
    
    for group in groups:
        if len(group) == 1:
            kept_notes.extend(group)
            continue
            
        group_sorted = sorted(group, key=lambda n: n["pitch"])
        to_remove = set()
        
        for i, base_n in enumerate(group_sorted):
            if id(base_n) in to_remove:
                continue
                
            base_p = base_n["pitch"]
            base_v = base_n.get("velocity", 0.5)
            
            for j in range(i+1, len(group_sorted)):
                over_n = group_sorted[j]
                if id(over_n) in to_remove:
                    continue
                    
                over_p = over_n["pitch"]
                diff = over_p - base_p
                
                # +12 (octave), +19 (octave+fifth), +24 (2 octaves)
                if diff in (12, 19, 24):
                    over_v = over_n.get("velocity", 0.5)
                    over_d = over_n["end"] - over_n["start"]
                    
                    # If overtone is softer or much shorter, remove it
                    if over_v < base_v * 1.1 or over_d < 0.15:
                        to_remove.add(id(over_n))
                        logs.append({
                            "pass": 2,
                            "action": "delete",
                            "note": dict(over_n),
                            "reason": f"Overtone of {base_p} (+{diff})"
                        })
                        
import math
from typing import List, Tuple, Dict, Any
from chord_theory import _parse_chord_name, _get_chord_notes_pc

# Pitch classes (0=C, 1=C#, ..., 11=B)
KEY_SIGNATURES = {
    "C":  [0, 2, 4, 5, 7, 9, 11],
    "G":  [0, 2, 4, 6, 7, 9, 11],
    "D":  [1, 2, 4, 6, 7, 9, 11],
    "A":  [1, 2, 4, 6, 8, 9, 11],
    "E":  [1, 3, 4, 6, 8, 9, 11],
    "B":  [1, 3, 4, 6, 8, 10, 11],
    "F":  [0, 2, 4, 5, 7, 9, 10],
    "Bb": [0, 2, 3, 5, 7, 9, 10],
    "Eb": [0, 2, 3, 5, 7, 8, 10],
    "Ab": [0, 1, 3, 5, 7, 8, 10],
    "Am": [0, 2, 4, 5, 7, 9, 11],
    "Em": [0, 2, 4, 6, 7, 9, 11],
    "Dm": [0, 2, 3, 5, 7, 9, 10],
    "Bm": [1, 2, 4, 6, 7, 9, 11],
}

def _group_simultaneous_notes(notes: List[dict], time_tolerance: float = 0.05) -> List[List[dict]]:
    """Group notes that start at roughly the same time (chords)."""
    if not notes:
        return []
    
    sorted_notes = sorted(notes, key=lambda n: n["start"])
    groups = []
    current_group = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        if note["start"] - current_group[0]["start"] <= time_tolerance:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)
    return groups

def _get_chord_at_time(chords: List[dict], t_sec: float) -> str:
    if not chords:
        return "N.C."
    for c in chords:
        start = c.get('start', 0.0)
        end = c.get('end', 9999.0)
        if start <= t_sec < end:
            return c.get('chord', "N.C.")
    return "N.C."

def _pass1_melodic_smoothing(notes: List[dict], logs: List[Dict[str, Any]]) -> List[dict]:
    """
    Pass 1: Octave error correction (Melodic Smoothing)
    Only applies to single-note melodic lines.
    Excludes base notes (E3 / MIDI 52 and below).
    """
    if not notes:
        return notes
        
    sorted_notes = sorted(notes, key=lambda n: n["start"])
    groups = _group_simultaneous_notes(sorted_notes, time_tolerance=0.08)
    
    for i in range(1, len(groups) - 1):
        prev_g = groups[i-1]
        curr_g = groups[i]
        next_g = groups[i+1]
        
        # Only apply to single-note melodies (not chords)
        if len(prev_g) == 1 and len(curr_g) == 1 and len(next_g) == 1:
            prev_n = prev_g[0]
            curr_n = curr_g[0]
            next_n = next_g[0]
            
            p1 = prev_n["pitch"]
            p2 = curr_n["pitch"]
            p3 = next_n["pitch"]
            
            # Exclude base notes and notes following base notes
            if p1 < 52 or p2 < 52 or p3 < 52:
                continue
            
            jump1 = p2 - p1
            jump2 = p3 - p2
            
            # Check for octave jumps (+/- 12 semitones roughly)
            if 10 <= abs(jump1) <= 14:
                # If we shift p2 by -octave, does it smooth the line?
                sign = -1 if jump1 > 0 else 1
                shifted_p2 = p2 + (sign * 12)
                
                # Check smoothness (stepwise or small leaps < 5 semitones)
                smooth_jump1 = abs(shifted_p2 - p1)
                smooth_jump2 = abs(p3 - shifted_p2)
                
                original_smoothness = abs(jump1) + abs(jump2)
                new_smoothness = smooth_jump1 + smooth_jump2
                
                # If shifting heavily improves smoothness and connects well
                if new_smoothness < original_smoothness and smooth_jump1 <= 5 and smooth_jump2 <= 5:
                    logs.append({
                        "pass": 1,
                        "action": "modify",
                        "note": dict(curr_n),
                        "reason": f"Melodic smoothing (p1={p1}, p2={p2}->{shifted_p2}, p3={p3})",
                        "modified_pitch": shifted_p2
                    })
                    curr_n["pitch"] = shifted_p2

    return sorted_notes

def _pass2_harmonic_merging(notes: List[dict], logs: List[Dict[str, Any]]) -> List[dict]:
    """
    Pass 2: Harmonic Merging (Resonance / Overtone deduplication)
    """
    groups = _group_simultaneous_notes(notes, time_tolerance=0.05)
    kept_notes = []
    
    for group in groups:
        if len(group) == 1:
            kept_notes.extend(group)
            continue
            
        group_sorted = sorted(group, key=lambda n: n["pitch"])
        to_remove = set()
        
        for i, base_n in enumerate(group_sorted):
            if id(base_n) in to_remove:
                continue
                
            base_p = base_n["pitch"]
            base_v = base_n.get("velocity", 0.5)
            
            for j in range(i+1, len(group_sorted)):
                over_n = group_sorted[j]
                if id(over_n) in to_remove:
                    continue
                    
                over_p = over_n["pitch"]
                diff = over_p - base_p
                
                # +12 (octave), +19 (octave+fifth), +24 (2 octaves)
                if diff in (12, 19, 24):
                    over_v = over_n.get("velocity", 0.5)
                    over_d = over_n["end"] - over_n["start"]
                    
                    # If overtone is softer or much shorter, remove it
                    if over_v < base_v * 1.1 or over_d < 0.15:
                        to_remove.add(id(over_n))
                        logs.append({
                            "pass": 2,
                            "action": "delete",
                            "note": dict(over_n),
                            "reason": f"Overtone of {base_p} (+{diff})"
                        })
                        
        for n in group:
            if id(n) not in to_remove:
                kept_notes.append(n)
                
    return kept_notes

def _pass3_non_chord_tone_filtering(notes: List[dict], chords: List[dict], key_sig: str, genre: str, logs: List[Dict[str, Any]]) -> List[dict]:
    """
    Pass 3: Non-Chord Tone Filtering (Ghost note removal)
    Genre Guard: For Jazz tracks, strict velocity thresholds and tension note whitelists.
    For unknown genres, strictly velocity <= min AND duration < 80ms.
    """
    groups = _group_simultaneous_notes(notes, time_tolerance=0.05)
    kept_notes = []
    
    allowed_pcs = KEY_SIGNATURES.get(key_sig, KEY_SIGNATURES["C"])
    is_jazz = genre.lower() == "jazz"
    is_unknown = not is_jazz # We can expand this later
    
    for group in groups:
        if len(group) <= 1:
            kept_notes.extend(group)
            continue
            
        group_sorted = sorted(group, key=lambda n: n["pitch"])
        root_p = group_sorted[0]["pitch"]
        
        # Get current chord
        t_sec = group_sorted[0]["start"]
        active_chord = _get_chord_at_time(chords, t_sec)
        
        chord_pcs = []
        if active_chord not in ("N.C.", "N", "X"):
            c_root, c_q = _parse_chord_name(active_chord)
            if c_root >= 0:
                chord_pcs = _get_chord_notes_pc(c_root, c_q)
        
        to_remove = set()
        min_vel = min(n.get("velocity", 0.5) for n in group)
        
        for n in group_sorted[1:]: # Don't remove the root
            p = n["pitch"]
            diff = p - root_p
            pc = p % 12
            interval = diff % 12
            
            is_tension_or_approach = False
            if is_jazz:
                if diff in (5, 14, 17, 21):
                    is_tension_or_approach = True
                else:
                    for c_pc in chord_pcs:
                        if (pc - c_pc) % 12 in (1, 11): # +/- 1 semitone
                            is_tension_or_approach = True
                            break
            
            is_dissonant = interval in (1, 6, 11)
            is_out_of_key = pc not in allowed_pcs and pc not in chord_pcs
            
            if is_dissonant or is_out_of_key:
                if not is_tension_or_approach:
                    v = n.get("velocity", 0.5)
                    d = n["end"] - n["start"]
                    
                    if is_jazz:
                        if (v <= min_vel * 1.1) and (d < 0.08):
                            to_remove.add(id(n))
                            logs.append({
                                "pass": 3,
                                "action": "delete",
                                "note": dict(n),
                                "reason": f"Dissonant in Jazz (int={interval}, v={v:.2f}, d={d:.2f})"
                            })
                    else: # Unknown genre guard
                        if (v <= min_vel * 1.2) and (d < 0.08):
                            to_remove.add(id(n))
                            logs.append({
                                "pass": 3,
                                "action": "delete",
                                "note": dict(n),
                                "reason": f"Dissonant in Unknown (int={interval}, v={v:.2f}, d={d:.2f})"
                            })
                            
        for n in group:
            if id(n) not in to_remove:
                kept_notes.append(n)
                
    return kept_notes

def heuristic_pitch_correction(notes: List[dict], chords: List[dict] = None, key: str = "C", genre: str = "unknown", dry_run: bool = False, verbose: bool = False) -> Tuple[List[dict], List[Dict[str, Any]]]:
    """
    Apply physical and musical heuristic rules to correct raw AI pitch predictions.
    Returns: (corrected_notes, logs)
    """
    import copy
    working_notes = copy.deepcopy(notes)
    if chords:
        pass
    else:
        chords = []
        
    logs = []
    
    # Pass 1
    working_notes = _pass1_melodic_smoothing(working_notes, logs)
    
    # Pass 2
    # working_notes = _pass2_harmonic_merging(working_notes, logs) # DISABLED due to >5% false deletion
    
    # Pass 3
    working_notes = _pass3_non_chord_tone_filtering(working_notes, chords, key, genre, logs)
    
    if verbose:
        p1 = sum(1 for l in logs if l['pass'] == 1)
        p2 = sum(1 for l in logs if l['pass'] == 2)
        p3 = sum(1 for l in logs if l['pass'] == 3)
        print(f"[Heuristic] Octave smoothed: {p1}, Harmonics merged: {p2}, Ghost notes filtered: {p3}")
        
    if dry_run:
        return notes, logs
    else:
        return working_notes, logs
