import json, numpy as np

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

# Filter pred to first 20 bars
pred = [n for n in pred if n.get('bar', 0) <= 20]

gt_notes = []
if 'measures_detailed' in gt:
    for m in gt['measures_detailed']:
        for r in m.get('notes', []):
            try:
                p = int(r.get('pitch', r.get('midi', -1)))
                if p >= 0: gt_notes.append({'pitch': p, 'string': int(r.get('string',0)), 'fret': int(r.get('fret',0))})
            except: pass

gt_p = [n['pitch'] for n in gt_notes]
pred_p = [int(n['pitch']) for n in pred]
matches = align_sequences(gt_p, pred_p)

print(f"Eval on measures 1-20 only:")
print(f"GT Notes: {len(gt_notes)}, Pred Notes: {len(pred)}")
print(f"Matched: {len(matches)}")
print(f"Extra: {len(pred) - len(matches)}")
print(f"Missed: {len(gt_notes) - len(matches)}")

simul = 0
for gi, pi in matches:
    if gt_notes[gi]['string'] == pred[pi]['string'] and gt_notes[gi]['fret'] == pred[pi]['fret']:
        simul += 1
print(f"String&Fret Exact: {100.0 * simul / len(matches):.2f}% ({simul}/{len(matches)})")
