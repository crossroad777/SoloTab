import json, os, sys, glob, numpy as np
from collections import Counter

gt_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
with open(gt_path, 'r', encoding='utf-8') as f:
    gt = json.load(f)

sessions = sorted(glob.glob(r'd:\Music\nextchord-solotab\uploads\20260322-*'))
if not sessions:
    print("No 20260322 sessions found!"); sys.exit(1)
latest = sessions[-1]
print(f"Latest session: {os.path.basename(latest)}")

notes_path = os.path.join(latest, 'notes_assigned.json')
if not os.path.exists(notes_path):
    notes_path = os.path.join(latest, 'notes.json')
with open(notes_path, 'r', encoding='utf-8') as f:
    notes_data = json.load(f)
detected = notes_data if isinstance(notes_data, list) else notes_data.get('notes', [])

beats_path = os.path.join(latest, 'beats.json')
with open(beats_path, 'r', encoding='utf-8') as f:
    beat_data = json.load(f)
beats = np.array(beat_data['beats'])
bpm = beat_data['bpm']

gt_measures = gt['measures_detailed']
names_map = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def pn(p): return f"{names_map[p%12]}{p//12-1}"

def get_measure(t):
    idx = int(np.searchsorted(beats, t))
    idx = max(0, min(idx, len(beats)-1))
    if idx > 0 and abs(beats[idx-1] - t) < abs(beats[idx] - t): idx -= 1
    return idx // 3

print(f"Detected total: {len(detected)} notes")
print(f"BPM: {bpm} (GT: {gt['metadata']['bpm']})")
print(f"Time: {beat_data.get('time_signature','?')} (GT: {gt['metadata']['time_signature']})")

print("\n" + "="*70)
print(" MEASURE-BY-MEASURE (A section: m1-13)")
print("="*70)

tot_gt = tot_match = tot_det = tot_fp = 0
str_ok = fret_ok = checked = 0

for gt_m in gt_measures[:13]:
    mn = gt_m['measure']
    gn = gt_m['notes']
    gp = [n['pitch'] for n in gn]
    dm = [n for n in detected if get_measure(float(n['start'])) == mn - 1]
    dp = [n['pitch'] for n in dm]
    gc, dc = Counter(gp), Counter(dp)
    matched = sum(min(gc[p], dc.get(p,0)) for p in gc)
    missed = len(gn) - matched
    fp = max(0, len(dm) - matched)
    tot_gt += len(gn); tot_match += matched; tot_det += len(dm); tot_fp += fp
    st = "OK" if matched >= len(gn)*0.7 else "WARN" if matched >= len(gn)*0.4 else "MISS"
    print(f"  M{mn:2d} {gt_m['chord']:>3s} | GT:{len(gn):2d} Det:{len(dm):2d} Match:{matched}/{len(gn)} [{st}]")
    if missed > 0:
        ms = [pn(p) for p in gc for _ in range(gc[p]-dc.get(p,0)) if gc[p]>dc.get(p,0)]
        print(f"       Missing: {', '.join(ms)}")
    # String/fret check
    dm2 = list(dm)
    for gn2 in gn:
        for d in dm2:
            if d['pitch'] == gn2['pitch']:
                checked += 1
                if d.get('string') == gn2['string']: str_ok += 1
                if d.get('fret') == gn2['fret']: fret_ok += 1
                dm2.remove(d); break

P = tot_match/tot_det if tot_det else 0
R = tot_match/tot_gt if tot_gt else 0
F1 = 2*P*R/(P+R) if (P+R) else 0

print(f"\n{'='*70}")
print(f" SUMMARY (measures 1-13)")
print(f"{'='*70}")
print(f"  GT notes:    {tot_gt}")
print(f"  Detected:    {tot_det}")
print(f"  Matched:     {tot_match}")
print(f"  Precision:   {P:.1%}")
print(f"  Recall:      {R:.1%}")
print(f"  F1 Score:    {F1:.1%}")
if checked:
    print(f"  String acc:  {str_ok}/{checked} = {str_ok/checked:.1%}")
    print(f"  Fret acc:    {fret_ok}/{checked} = {fret_ok/checked:.1%}")
