import os
import json
import glob
import sys
import copy
from collections import defaultdict

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from finger_assigner import assign_fingers
from e2e_pipeline_benchmark import TARGET_TRACKS, ANNOTATIONS_DIR, load_jams_notes_with_string

def get_phrase_groups(notes, gap=0.5):
    """0.5秒以上の間隔でノートをフレーズに分割"""
    notes = sorted(notes, key=lambda x: x.get('start', x.get('start_time', 0.0)))
    phrases = []
    current_phrase = []
    
    for n in notes:
        start = n.get('start', n.get('start_time', 0.0))
        if not current_phrase:
            current_phrase.append(n)
        else:
            prev = current_phrase[-1]
            prev_end = prev.get('end', prev.get('end_time', 0.0))
            if start - prev_end >= gap:
                phrases.append(current_phrase)
                current_phrase = [n]
            else:
                current_phrase.append(n)
    if current_phrase:
        phrases.append(current_phrase)
    return phrases

def evaluate_fingering(notes, track_name, is_gt=False):
    metrics = {
        'total_notes': len(notes),
        'position_shifts': 0,
        'violation_cross': 0,
        'violation_span': 0,
        'violation_order': 0,
        'violation_stretch': 0,
        'total_chords': 0,
        'barre_opportunities': 0,
        'barre_success': 0,
        'scale_run_opportunities': 0,
        'scale_run_success': 0,
        'violations': [] # (note, reason)
    }
    
    if not notes:
        return metrics

    phrases = get_phrase_groups(notes)
    
    # 1. Position Consistency
    for phrase in phrases:
        phrase_positions = []
        for n in phrase:
            f = n.get('fret', 0)
            fing = n.get('finger', 0)
            if f > 0 and fing in [1, 2, 3, 4]:
                implied_pos = f - fing + 1
                phrase_positions.append(implied_pos)
                
        shifts = 0
        for i in range(1, len(phrase_positions)):
            if abs(phrase_positions[i] - phrase_positions[i-1]) > 1:
                shifts += 1
        metrics['position_shifts'] += shifts

    # 2. Biomechanical Violations
    chords = defaultdict(list)
    for n in notes:
        start = n.get('start', n.get('start_time', 0.0))
        chord_key = round(start, 2)
        chords[chord_key].append(n)

    for c_time, chord in chords.items():
        if len(chord) > 1:
            metrics['total_chords'] += 1
            
            # --- Maximum Stretch Violation Rate (New) ---
            # Any chord with fret difference > 7 is a violation
            all_frets = [n.get('fret', 0) for n in chord if n.get('fret', 0) > 0]
            if all_frets and max(all_frets) - min(all_frets) > 7:
                metrics['violation_stretch'] += 1
                metrics['violations'].append((chord[0], f"Max Stretch > 7 ({min(all_frets)}f to {max(all_frets)}f)"))

            # --- Span > 6 (Fingered notes only) ---
            fingered_frets = [n.get('fret', 0) for n in chord if n.get('fret', 0) > 0 and n.get('finger', 0) in [1,2,3,4]]
            if fingered_frets and max(fingered_frets) - min(fingered_frets) > 6:
                metrics['violation_span'] += len(chord)
                metrics['violations'].append((chord[0], f"Fingered Span > 6"))
            
            # --- Finger Order / Cross ---
            for i in range(len(chord)):
                for j in range(i+1, len(chord)):
                    n1 = chord[i]
                    n2 = chord[j]
                    f1, fing1 = n1.get('fret', 0), n1.get('finger', 0)
                    f2, fing2 = n2.get('fret', 0), n2.get('finger', 0)
                    
                    if f1 > 0 and f2 > 0 and fing1 in [1,2,3,4] and fing2 in [1,2,3,4]:
                        if fing1 < fing2 and f1 > f2:
                            metrics['violation_order'] += 2
                            metrics['violations'].append((n1, "Cross/Order"))
                        elif fing1 > fing2 and f1 < f2:
                            metrics['violation_order'] += 2
                            metrics['violations'].append((n1, "Cross/Order"))

        # 3. Barre Detection
        fret_groups = defaultdict(list)
        for n in chord:
            if n.get('fret', 0) > 0:
                fret_groups[n.get('fret', 0)].append(n)
        
        for f, grp in fret_groups.items():
            if len(grp) >= 2:
                metrics['barre_opportunities'] += 1
                fingers = set([n.get('finger', 0) for n in grp])
                if len(fingers) == 1 and 1 in fingers:
                    metrics['barre_success'] += 1
                elif len(fingers) == 1 and list(fingers)[0] in [2,3,4]:
                     metrics['barre_success'] += 1

    # 4. Scale Run Finger Order
    notes_sorted = sorted(notes, key=lambda x: x.get('start', x.get('start_time', 0.0)))
    run = []
    for n in notes_sorted:
        if not run:
            run.append(n)
        else:
            prev = run[-1]
            gap = n.get('start', n.get('start_time', 0.0)) - prev.get('end', prev.get('end_time', 0.0))
            if n.get('string') == prev.get('string') and gap < 0.5 and n.get('fret', 0) > 0 and prev.get('fret', 0) > 0:
                run.append(n)
            else:
                if len(run) >= 3:
                    _eval_run(run, metrics)
                run = [n]
    if len(run) >= 3:
        _eval_run(run, metrics)

    return metrics

def _eval_run(run, metrics):
    metrics['scale_run_opportunities'] += 1
    # Check if pitches are strictly ascending or descending
    frets = [n.get('fret', 0) for n in run]
    fingers = [n.get('finger', 0) for n in run]
    
    is_asc = all(frets[i] < frets[i+1] for i in range(len(frets)-1))
    is_desc = all(frets[i] > frets[i+1] for i in range(len(frets)-1))
    
    if is_asc:
        # fingers should be ascending
        valid = True
        for i in range(len(fingers)-1):
            if fingers[i] in [1,2,3,4] and fingers[i+1] in [1,2,3,4]:
                if fingers[i] >= fingers[i+1]:
                    valid = False
        if valid: metrics['scale_run_success'] += 1
    elif is_desc:
        valid = True
        for i in range(len(fingers)-1):
            if fingers[i] in [1,2,3,4] and fingers[i+1] in [1,2,3,4]:
                if fingers[i] <= fingers[i+1]:
                    valid = False
        if valid: metrics['scale_run_success'] += 1
    else:
        # Not a strict scale run, ignore
        metrics['scale_run_opportunities'] -= 1


def main():
    import time
    import tempfile
    import shutil
    from pathlib import Path
    try:
        from pipeline import run_pipeline
    except ImportError as e:
        print(f"Error importing pipeline: {e}")
        sys.exit(1)

    print("==================================================")
    print(" FINGERING QUALITY BENCHMARK (GuitarSet Mini)")
    print("==================================================")
    
    # TUNING for standard EADGBE
    TUNING_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

    results_e2e = {}
    results_gt = {}

    start_time = time.time()
    for i, track in enumerate(TARGET_TRACKS):
        print(f"\n[{i+1}/{len(TARGET_TRACKS)}] Evaluating Track: {track}")
        
        # --- 1. E2E Output Evaluation ---
        # Run pipeline to get notes_assigned_original.json
        temp_dir = tempfile.mkdtemp(prefix="solotab_e2e_")
        session_id = f"benchmark_{track}"
        wav_path = os.path.join(ANNOTATIONS_DIR.replace("annotation", "audio_mono-mic"), f"{track}_mic.wav")
        
        try:
            run_pipeline(
                session_id=session_id,
                session_dir=Path(temp_dir),
                wav_path=Path(wav_path),
                tuning_name="standard",
                skip_demucs=True,
                fast_moe=True
            )
            
            json_path = os.path.join(temp_dir, "notes_assigned_original.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(temp_dir, "notes_assigned.json")
                
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    e2e_notes = json.load(f)
                e2e_metrics = evaluate_fingering(e2e_notes, track, is_gt=False)
                results_e2e[track] = e2e_metrics
            else:
                print(f"  Warning: E2E notes not found for {track}")
        except Exception as e:
            print(f"  Error running E2E pipeline for {track}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # --- 2. GT Pitch Input Evaluation ---
        jams_path = os.path.join(ANNOTATIONS_DIR, f"{track}.jams")
        if os.path.exists(jams_path):
            gt_notes = load_jams_notes_with_string(jams_path)
            # Calculate frets based on GT pitch and GT string
            valid_gt_notes = []
            for n in gt_notes:
                string = n.get('string')
                pitch = n.get('pitch')
                if string in TUNING_MIDI:
                    fret = round(pitch - TUNING_MIDI[string])
                    if fret >= 0:
                        n['fret'] = fret
                        n['start_time'] = n['start']
                        n['end_time'] = n['end']
                        valid_gt_notes.append(n)
            
            # Pass to finger_assigner (must mock basic structure)
            # finger_assigner expects notes to be sorted
            valid_gt_notes.sort(key=lambda x: x['start_time'])
            
            try:
                assigned_gt_notes = assign_fingers(valid_gt_notes)
                gt_metrics = evaluate_fingering(assigned_gt_notes, track, is_gt=True)
                results_gt[track] = gt_metrics
            except Exception as e:
                print(f"  Error assigning GT fingers for {track}: {e}")
        else:
            print(f"  Warning: JAMS not found for {track}")

    # --- Print Summary ---
    def print_summary(res, label):
        if not res: return
        t_notes = sum(m['total_notes'] for m in res.values())
        t_chords = sum(m['total_chords'] for m in res.values())
        t_shifts = sum(m['position_shifts'] for m in res.values())
        t_viols = sum(m['violation_cross'] + m['violation_span'] + m['violation_order'] for m in res.values())
        t_stretch = sum(m['violation_stretch'] for m in res.values())
        t_b_opps = sum(m['barre_opportunities'] for m in res.values())
        t_b_succ = sum(m['barre_success'] for m in res.values())
        t_s_opps = sum(m['scale_run_opportunities'] for m in res.values())
        t_s_succ = sum(m['scale_run_success'] for m in res.values())

        pos_consistency = 1.0 - (t_shifts / t_notes) if t_notes > 0 else 0
        viol_rate = t_viols / t_notes if t_notes > 0 else 0
        stretch_rate = t_stretch / t_chords if t_chords > 0 else 0
        barre_rate = t_b_succ / t_b_opps if t_b_opps > 0 else 0
        scale_rate = t_s_succ / t_s_opps if t_s_opps > 0 else 0

        print(f"\n[{label} SUMMARY]")
        print(f"  Position Consistency Score   : {pos_consistency:.4f} (Target: >= 0.85)")
        print(f"  Biomechanical Violation Rate : {viol_rate:.4f} (Target: 0.0)")
        print(f"  Max Stretch Violation Rate   : {stretch_rate:.4f} (Target: 0.0)")
        print(f"  Barre Detection Rate         : {barre_rate:.4f} (Target: >= 0.80)")
        print(f"  Scale Run Finger Order Rate  : {scale_rate:.4f} (Target: >= 0.90)")
        
        # Track by track consistency
        print(f"\n  [Track-by-Track Position Consistency]")
        for track, m in res.items():
            n = m['total_notes']
            s = m['position_shifts']
            c = 1.0 - (s/n) if n > 0 else 0
            v = m['violation_cross'] + m['violation_span'] + m['violation_order'] + m['violation_stretch']
            print(f"    {track:<25}: Score={c:.4f}, Violations={v}")
            if v > 0:
                for viol in m['violations'][:3]:
                    nt, reason = viol
                    print(f"      -> {reason} at {nt.get('start', nt.get('start_time', 0)):.2f}s (String {nt.get('string')}, Fret {nt.get('fret')}, Finger {nt.get('finger')})")

    print_summary(results_e2e, "E2E PIPELINE")
    print_summary(results_gt, "GT PITCH INPUT")

if __name__ == "__main__":
    main()
