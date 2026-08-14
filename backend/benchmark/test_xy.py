import json, collections
import numpy as np

def align_sequences(gt_pitches, pred_pitches):
    n, m = len(gt_pitches), len(pred_pitches)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_pitches[i-1] == pred_pitches[j-1] else 2
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    i, j = n, m
    matches = []
    while i > 0 and j > 0:
        cost = 0 if gt_pitches[i-1] == pred_pitches[j-1] else 2
        if dp[i][j] == dp[i-1][j-1] + cost:
            if cost == 0: matches.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + 1: i -= 1
        else: j -= 1
    return matches[::-1]

gt = json.load(open(r'backend/ground_truth/romance_forbidden_games.json', encoding='utf-8'))
pred = json.load(open(r'uploads/20260814-013518-2a384c/notes_assigned.json', encoding='utf-8'))
if isinstance(pred, dict) and 'notes' in pred: pred = pred['notes']

gt_notes = []
if 'measures_detailed' in gt:
    for m in gt['measures_detailed']:
        for r in m.get('notes', []):
            try:
                p = int(r.get('pitch', r.get('midi', -1)))
                if p >= 0: gt_notes.append({'pitch': p})
            except: pass

gt_p = [n['pitch'] for n in gt_notes]
pred_p = [int(n['pitch']) for n in pred]
matches = align_sequences(gt_p, pred_p)
matched_pred = {j for i,j in matches}

extra_indices = [j for j in range(len(pred_p)) if j not in matched_pred]
bars = [pred[j].get('bar', 0) for j in extra_indices]

print("--- Test X: Bar histogram for Extra 233 notes ---")
bars_1_20 = sum(1 for b in bars if b <= 20)
bars_21_49 = sum(1 for b in bars if b > 20)
print(f"  Bars 1-20: {bars_1_20} notes")
print(f"  Bars 21-49: {bars_21_49} notes")

print("\n--- Test Y: _tie_stop flag presence ---")
ties = sum(1 for n in pred if '_tie_stop' in n or '_tie_start' in n)
print(f"  Tied notes count: {ties}")
