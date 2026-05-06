# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Romance de Amor - full pipeline evaluation
"""
import os, json, copy
sys.path.insert(0, 'D:/Music/nextchord-solotab/backend')
sys.path.insert(0, 'D:/Music/nextchord-solotab/music-transcription/python')
os.chdir('D:/Music/nextchord-solotab/backend')

wav_path = 'D:/Music/nextchord-solotab/uploads/20260429-003136/converted.wav'
gt = json.load(open('D:/Music/nextchord-solotab/backend/reference_data/romance_ground_truth.json'))

bpm = 89.1
beat_interval = 60.0 / bpm
first_note_time = 1.904
ioi = beat_interval / 3.0

def get_expected_time(m_num, beat, role, string):
    m_start = first_note_time + (m_num - 1) * beat_interval * 3
    beat_start = m_start + (beat - 1) * beat_interval
    if role in ('bass', 'melody'):
        return beat_start
    elif role == 'inner':
        if string == 2: return beat_start + ioi
        elif string == 3: return beat_start + ioi * 2
    return beat_start

def evaluate(notes, label, tol=0.12, verbose=False):
    tc=0; tw=0; tm=0
    rs = {'melody': [0,0], 'inner': [0,0], 'bass': [0,0]}
    matched = set(); det = []
    for m in gt['measures']:
        mn = m['measure']
        for gn in m['notes']:
            et = get_expected_time(mn, gn['beat'], gn['role'], gn['string'])
            r = gn['role']; rs[r][0] += 1
            best=None; bd=float('inf'); bi=-1
            for i, n in enumerate(notes):
                if i in matched: continue
                if n['pitch'] == gn['pitch']:
                    d = abs(n['start'] - et)
                    if d < tol and d < bd: best=n; bd=d; bi=i
            if best:
                matched.add(bi); rs[r][1] += 1
                sf = best['string']==gn['string'] and best['fret']==gn['fret']
                if sf: tc += 1
                else: tw += 1
                if verbose:
                    c = " *" if best.get('_open_string_corrected') else ""
                    s = "OK" if sf else f"WRONG(s{best['string']}f{best['fret']})"
                    det.append(f"  M{mn}b{gn['beat']} {r:>6} {gn['note']:>3} s{gn['string']}f{gn['fret']} dt={bd:.3f} {s}{c}")
            else:
                tm += 1
                if verbose: det.append(f"  M{mn}b{gn['beat']} {r:>6} {gn['note']:>3} s{gn['string']}f{gn['fret']} MISSING")
    mn2 = [n for n in notes if 1.8<=n['start']<10.0]
    tgt = sum(v[0] for v in rs.values())
    dtot = sum(v[1] for v in rs.values())
    tp=dtot; fp=max(0,len(mn2)-dtot); fn=tm
    rec = tp/max(tp+fn,1)*100; pre = tp/max(tp+fp,1)*100
    f1 = 2*rec*pre/max(rec+pre,0.01)
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    for r in ('melody','inner','bass'):
        t,d = rs[r]; print(f"  {r:>7}: {d}/{t} ({d*100/max(t,1):.0f}%)")
    print(f"  Recall={rec:.1f}% Prec={pre:.1f}% F1={f1:.1f}%")
    print(f"  Correct={tc} Wrong={tw} Missing={tm} Extra={fp}")
    if verbose:
        for d in det: print(d)
    return {'recall':rec,'f1':f1,'correct':tc,'missing':tm,'wrong':tw}

# === Step 1: Raw ===
from guitar_transcriber import transcribe_guitar
raw = transcribe_guitar(wav_path)['notes']
r1 = evaluate(raw, "Step 1: Conformer raw")

# === Step 2: + Music Theory (chord refined + open string fix) ===
from music_theory import apply_music_theory_filter
from beat_detector import detect_beats
from chord_detector import detect_chords
from solotab_utils import STANDARD_TUNING

br = detect_beats(wav_path)
chords = detect_chords(wav_path, beats=br['beats'])
fixed, _ = apply_music_theory_filter(
    copy.deepcopy(raw), chords, tuning=STANDARD_TUNING, beats=br['beats']
)
r2 = evaluate(fixed, "Step 2: + Music Theory (chord refine + open string fix)", verbose=True)

print(f"\n{'='*65}")
print(f"  IMPROVEMENT")
print(f"{'='*65}")
print(f"  Recall:  {r1['recall']:.1f}% -> {r2['recall']:.1f}% ({r2['recall']-r1['recall']:+.1f})")
print(f"  F1:      {r1['f1']:.1f}% -> {r2['f1']:.1f}% ({r2['f1']-r1['f1']:+.1f})")
print(f"  Missing: {r1['missing']} -> {r2['missing']}")
print(f"  Wrong:   {r1['wrong']} -> {r2['wrong']}")
print("\nDone.")
