#!/usr/bin/env python3
"""
ablation_finger_postprocess.py
====================================================================
Step 2 分解診断: finger_assigner の後処理が CNN 予測を「直している」
のか「壊している」のかを、GP5 正解データに対して数値で確定させる。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os, sys, json, copy, argparse
from collections import defaultdict
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
import numpy as np
import finger_assigner as fa

STAGE_ORDER = ['cnn_only', 'after_chord', 'after_viterbi', 'after_templates', 'after_prox', 'after_pattern', 'after_pivot']
STAGE_LABEL = {'cnn_only': 'CNN only (baseline)', 'after_chord': '+ chord conflict', 'after_viterbi': '+ Viterbi DP', 'after_templates': '+ templates', 'after_prox': '+ pitch proximity', 'after_pattern': '+ pattern consistency', 'after_pivot': '+ pivot fingers'}

def _extract_finger(note):
    for k in ('finger', 'left_hand_finger', 'fingering', 'leftHandFingering', 'lh_finger', 'label'):
        if k in note and note[k] is not None: return int(note[k])
    return None

def _normalize_note(raw):
    try:
        pitch = int(raw.get('pitch', raw.get('midi', -1)))
        string = int(raw.get('string', 0))
        fret = int(raw.get('fret', raw.get('value', -1)))
        finger = _extract_finger(raw)
        start = float(raw.get('start', raw.get('start_time', raw.get('time', raw.get('onset', 0.0)))))
        duration = float(raw.get('duration', raw.get('dur', raw.get('end', start + 0.3) - start if 'end' in raw else 0.3)))
    except (TypeError, ValueError): return None
    if fret == 0: finger = 0
    if pitch < 0 or string < 1 or string > 6 or fret < 0 or finger is None: return None
    return {'pitch': pitch, 'string': string, 'fret': fret, '_gt_finger': finger, 'start': start, 'duration': duration}

def load_ground_truth(path):
    songs = []
    ext = os.path.splitext(path)[1].lower()
    if ext == '.jsonl':
        by_song = defaultdict(list)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                n = _normalize_note(json.loads(line))
                if n is not None: by_song[json.loads(line).get('file', '_all')].append(n)
        songs = [sorted(v, key=lambda x: x['start']) for v in by_song.values()]
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        if isinstance(data, dict) and 'songs' in data: data = data['songs']
        elif isinstance(data, dict) and 'phrases' in data: data = data['phrases']
        elif isinstance(data, dict) and 'measures_detailed' in data:
            notes_flat = []
            for m in data['measures_detailed']: notes_flat.extend(m.get('notes', []))
            notes = [n for n in [_normalize_note(r) for r in notes_flat] if n is not None]
            if notes: songs.append(sorted(notes, key=lambda x: x['start']))
            return songs
            
        if isinstance(data, list) and data and isinstance(data[0], list):
            for song in data:
                notes = [n for n in [_normalize_note(r) for r in song] if n is not None]
                if notes: songs.append(sorted(notes, key=lambda x: x['start']))
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            if 'notes' in data[0]:
                for phrase in data:
                    notes = [n for n in [_normalize_note(r) for r in phrase['notes']] if n is not None]
                    if notes: songs.append(sorted(notes, key=lambda x: x['start']))
            else:
                by_song = defaultdict(list)
                for raw in data:
                    n = _normalize_note(raw)
                    if n is not None: by_song[raw.get('file', '_all')].append(n)
                songs = [sorted(v, key=lambda x: x['start']) for v in by_song.values()]
    return [s for s in songs if len(s) >= 4]

def _group_chords(sorted_notes, threshold=0.03):
    if not sorted_notes: return []
    groups = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if note.get('start', 0) - groups[-1][0].get('start', 0) <= threshold: groups[-1].append(note)
        else: groups.append([note])
    return groups

def _snapshot(notes): return [int(n.get('left_hand_finger', -1)) for n in notes]

def run_staged_pipeline(notes, detected_key=None, skip_pattern=False, skip_prox=False, skip_pivot=False):
    results = {}
    tempo = fa._estimate_tempo(notes)
    for note in notes: note['_estimated_tempo'] = tempo; note.setdefault('_technique', note.get('technique'))
    cnn_results = fa._cnn_predict(notes)
    use_cnn = cnn_results is not None
    for i, note in enumerate(notes):
        fret = note.get('fret', 0) or 0
        if not isinstance(fret, (int, float)): fret = 0
        if fret == 0: note['left_hand_finger'] = 0; note['_finger_conf'] = 1.0; continue
        if use_cnn:
            pred, probs = cnn_results[i]
            note['_finger_probs'] = probs
            if fa._is_valid_finger(fret, pred): note['left_hand_finger'] = pred; note['_finger_conf'] = float(probs[pred])
            else:
                order = np.argsort(-probs); assigned = False
                for alt in order:
                    alt = int(alt)
                    if fa._is_valid_finger(fret, alt): note['left_hand_finger'] = alt; note['_finger_conf'] = float(probs[alt]); assigned = True; break
                if not assigned: note['left_hand_finger'] = fa._pdmx_predict(note.get('string', 3), fret); note['_finger_conf'] = 0.5
        else: note['left_hand_finger'] = fa._pdmx_predict(note.get('string', 3), fret); note['_finger_conf'] = 0.5
    results['cnn_only'] = _snapshot(notes)
    groups = _group_chords(sorted(notes, key=lambda n: n.get('start', 0)))
    fa._mark_bend_support_context(groups)
    for group in groups:
        if len(group) > 1: fa._resolve_chord_conflicts(group)
    results['after_chord'] = _snapshot(notes)
    fa._mark_anchor_context(notes); fa._propagate_barre_context(notes, groups); fa._propagate_chord_position(notes, groups)
    fa._viterbi_finger_dp(notes, phrase_gap=0.5); results['after_viterbi'] = _snapshot(notes)
    try: fa.apply_phrase_templates(notes)
    except Exception: pass
    results['after_templates'] = _snapshot(notes)
    
    if not skip_prox: fa._apply_pitch_proximity_rule(notes)
    results['after_prox'] = _snapshot(notes)
    
    if not skip_pattern: fa._enforce_pattern_consistency(notes)
    results['after_pattern'] = _snapshot(notes)
    
    if not skip_pivot: fa._apply_pivot_fingers(notes)
    results['after_pivot'] = _snapshot(notes)
    return results, {'use_cnn': use_cnn}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-songs', type=int, default=0)
    ap.add_argument('--output', default=None)
    ap.add_argument('--skip-pattern', action='store_true')
    ap.add_argument('--skip-prox', action='store_true')
    ap.add_argument('--skip-pivot', action='store_true')
    args = ap.parse_args()
    songs = load_ground_truth(args.data)
    if args.max_songs > 0: songs = songs[:args.max_songs]
    total_notes = sum(len(s) for s in songs)
    acc = {s: {'n': 0, 'exact': 0, 'tol1': 0, 'cc': 0, 'cw': 0, 'wc': 0, 'ww': 0} for s in STAGE_ORDER}
    cnn_mode = None
    for si, song in enumerate(songs):
        notes = copy.deepcopy(song)
        gt = [int(n['_gt_finger']) for n in notes]
        stage_results, meta = run_staged_pipeline(notes, skip_pattern=args.skip_pattern, skip_prox=args.skip_prox, skip_pivot=args.skip_pivot)
        if cnn_mode is None: cnn_mode = 'CNN' if meta['use_cnn'] else 'PDMX'
        prev = None
        for stage in STAGE_ORDER:
            pred = stage_results[stage]
            a = acc[stage]
            a['n'] += len(gt)
            a['exact'] += sum(1 for g, p in zip(gt, pred) if g == p)
            a['tol1'] += sum(1 for g, p in zip(gt, pred) if abs(g - p) <= 1)
            if prev is not None:
                for g, p, pp in zip(gt, pred, prev):
                    if g == pp and g == p: a['cc'] += 1
                    elif g == pp and g != p: a['cw'] += 1
                    elif g != pp and g == p: a['wc'] += 1
                    else: a['ww'] += 1
            prev = pred
    print(f"\n{'-'*68}\n{'ステージ':<26}{'Acc%':>7}{'±1%':>7}{'net':>8}{'直した':>8}{'壊した':>8}")
    summary = {}
    for stage in STAGE_ORDER:
        a = acc[stage]
        if a['n'] == 0: continue
        accu, tol1 = 100.0 * a['exact'] / a['n'], 100.0 * a['tol1'] / a['n']
        net = a['wc'] - a['cw']
        ns, fs, bs = (f"{net:+d}", f"{a['wc']}", f"{a['cw']}") if stage != 'cnn_only' else ("-", "-", "-")
        print(f"{STAGE_LABEL[stage]:<26}{accu:>7.2f}{tol1:>7.2f}{ns:>8}{fs:>8}{bs:>8}")
        summary[stage] = {'accuracy': accu, 'tol1': tol1, 'net': net, 'fixed': a['wc'], 'broke': a['cw']}
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f: json.dump({'stages': summary}, f, indent=2)

if __name__ == '__main__': main()
