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
                t = float(r.get('start', r.get('start_time', 0.0)))
                if p >= 0: gt_notes.append({'pitch': p, 'start': t})
            except: pass

gt_p = [n['pitch'] for n in gt_notes]
pred_p = [int(n.get('pitch', -1)) for n in pred]
matches = align_sequences(gt_p, pred_p)
matched_pred = {j for i,j in matches}

extra_indices = [j for j in range(len(pred_p)) if j not in matched_pred and pred[j].get('bar', 0) <= 20]
extra_notes = [pred[j] for j in extra_indices]

print(f"Total extra notes in bars 1-20: {len(extra_notes)}")

cat_A = 0
cat_B = 0
cat_C = 0
cat_D = 0

for n in extra_notes:
    if n.get('_tie_stop'):
        cat_A += 1
        continue
    
    t = float(n.get('start', n.get('start_time', 0.0)))
    p = int(n.get('pitch', -1))
    
    # Find nearest GT note in time
    near_gt = [gn for gn in gt_notes if abs(gn['start'] - t) <= 0.08]
    
    if any(gn['pitch'] == p for gn in near_gt):
        cat_B += 1
        continue
        
    if any(gn['pitch'] in [p + 12, p - 12, p + 7, p - 7] for gn in near_gt):
        cat_C += 1
        continue
        
    cat_D += 1

print(f"Category A (tie artifact):  {cat_A} 音")
print(f"Category B (same-pitch):   {cat_B} 音")
print(f"Category C (harmonic):     {cat_C} 音")
print(f"Category D (other):        {cat_D} 音")
