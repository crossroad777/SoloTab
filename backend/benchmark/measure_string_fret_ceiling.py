#!/usr/bin/env python3
import sys, os, json, argparse
import numpy as np
from collections import defaultdict

def _normalize_gt_note(raw):
    try:
        pitch = int(raw.get('pitch', raw.get('midi', -1)))
        string = int(raw.get('string', 0))
        fret = int(raw.get('fret', raw.get('value', -1)))
    except: return None
    if pitch < 0 or string < 1 or string > 6 or fret < 0: return None
    return {'pitch': pitch, 'string': string, 'fret': fret}

def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
    notes = []
    if 'measures_detailed' in data:
        for m in data['measures_detailed']:
            for r in m.get('notes', []):
                n = _normalize_gt_note(r)
                if n: notes.append(n)
    return notes

def load_output(path):
    with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
    if isinstance(data, dict) and 'notes' in data: data = data['notes']
    notes = []
    for r in data:
        n = _normalize_gt_note(r)
        if n: notes.append(n)
    return notes

def align_sequences(gt_pitches, pred_pitches):
    n, m = len(gt_pitches), len(pred_pitches)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_pitches[i-1] == pred_pitches[j-1] else 2
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion (missed)
                           dp[i][j-1] + 1,      # insertion (extra)
                           dp[i-1][j-1] + cost) # match/replace
                           
    i, j = n, m
    matches = []
    while i > 0 and j > 0:
        cost = 0 if gt_pitches[i-1] == pred_pitches[j-1] else 2
        if dp[i][j] == dp[i-1][j-1] + cost:
            if cost == 0:
                matches.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matches[::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', required=True)
    args = ap.parse_args()

    gt_notes = load_ground_truth(args.gt)
    pred_notes = load_output(args.pred)
    
    gt_pitches = [n['pitch'] for n in gt_notes]
    pred_pitches = [n['pitch'] for n in pred_notes]
    
    matches = align_sequences(gt_pitches, pred_pitches)
    missed = len(gt_notes) - len(matches)
    extra = len(pred_notes) - len(matches)
    
    print("=" * 68)
    print("STRING / FRET CEILING MEASUREMENT (Phase A)")
    print(f"GT Notes: {len(gt_notes)} | Pred Notes: {len(pred_notes)}")
    print("=" * 68)
    print(f"[Alignment] Matched: {len(matches)}")
    print(f"[Alignment] Missed (in GT, no Pred): {missed} ({100*missed/len(gt_notes):.1f}%)")
    print(f"[Alignment] Extra (in Pred, no GT): {extra} ({100*extra/len(pred_notes):.1f}%)")
    
    if len(matches) == 0:
        print("Error: No matches found. Alignment failed.")
        return
        
    string_exact = fret_exact = fret_tol1 = simultaneous = 0
    error_dist_string = defaultdict(int)
    error_dist_fret = defaultdict(int)
    error_examples = []
    
    for gi, pi in matches:
        g, p = gt_notes[gi], pred_notes[pi]
        s_match, f_match = (g['string'] == p['string']), (g['fret'] == p['fret'])
        
        if s_match: string_exact += 1
        else: error_dist_string[g['string']] += 1
            
        if f_match: fret_exact += 1
        elif not s_match:
            fret_bin = (g['fret'] // 5) * 5
            error_dist_fret[fret_bin] += 1
            
        if abs(g['fret'] - p['fret']) <= 1: fret_tol1 += 1
        if s_match and f_match: simultaneous += 1
        else:
            error_examples.append({
                'pitch': g['pitch'],
                'gt_s': g['string'], 'gt_f': g['fret'],
                'pred_s': p['string'], 'pred_f': p['fret']
            })
            
    print("\n" + "=" * 68)
    print("CEILING ACCURACY (Matched Notes Only)")
    print("-" * 68)
    N = len(matches)
    print(f"String Exact   : {100.0 * string_exact / N:.2f}% ({string_exact}/{N})")
    print(f"Fret Exact     : {100.0 * fret_exact / N:.2f}% ({fret_exact}/{N})")
    print(f"Fret ±1        : {100.0 * fret_tol1 / N:.2f}% ({fret_tol1}/{N})")
    print(f"String & Fret  : {100.0 * simultaneous / N:.2f}% ({simultaneous}/{N})  <-- 運指の実質上限")
    
    print("\n" + "=" * 68)
    print("ERROR DISTRIBUTION (String/Fret mismatches)")
    print("-" * 68)
    print("Missed String by GT String:")
    for s in range(1, 7): print(f"  String {s}: {error_dist_string[s]} errors")
    print("\nMissed String/Fret by GT Fret Bracket:")
    for f in sorted(error_dist_fret.keys()): print(f"  Fret {f}-{f+4}: {error_dist_fret[f]} errors")
    
    print("\n" + "=" * 68)
    print("CONCRETE ERROR EXAMPLES (up to 10)")
    print("-" * 68)
    print(f"{'Pitch':<7} | {'GT String/Fret':<15} | {'Pred String/Fret':<15}")
    for ex in error_examples[:10]:
        print(f"{ex['pitch']:<7} | {ex['gt_s']}弦 / {ex['gt_f']:<2}fret       | {ex['pred_s']}弦 / {ex['pred_f']:<2}fret")
    print("=" * 68)

if __name__ == '__main__':
    main()
