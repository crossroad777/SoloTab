import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
with open('D:/Music/nextchord-solotab/uploads/20260429-113238/notes.json') as f:
    raw = json.load(f).get('notes', [])
with open('D:/Music/nextchord-solotab/uploads/20260429-113238/beats.json') as f:
    beats = json.load(f)['beats']

measures = []
for i in range(0, len(beats)-2, 3):
    measures.append((beats[i], beats[i+3] if i+3 < len(beats) else beats[-1]+0.67))

def get_melody(m_start, m_end, notes):
    s1 = sorted([n for n in notes if m_start <= n['start'] < m_end and n.get('string') == 1], key=lambda n: n['start'])
    pitches = []
    for b in range(3):
        bs = m_start + (m_end-m_start)*b/3
        be = m_start + (m_end-m_start)*(b+1)/3
        bn = [n for n in s1 if bs <= n['start'] < be]
        pitches.append(bn[0]['pitch'] if bn else None)
    return pitches

print("=== Melody pitch: Em section vs E major section (MoE raw) ===")
for mi in range(16):
    minor_m = mi
    major_m = mi + 16
    if minor_m < len(measures) and major_m < len(measures):
        minor_mel = get_melody(measures[minor_m][0], measures[minor_m][1], raw)
        major_mel = get_melody(measures[major_m][0], measures[major_m][1], raw)
        diffs = []
        for a, b in zip(minor_mel, major_mel):
            if a is not None and b is not None:
                diffs.append(b - a)
            else:
                diffs.append(None)
        print(f"M{mi+1:>2}  Em={minor_mel}  Ema={major_mel}  diff={diffs}")
