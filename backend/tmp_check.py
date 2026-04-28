import json
n = json.load(open('D:/Music/nextchord-solotab/uploads/20260429-010733/notes.json'))
m1 = [x for x in n if 1.8 <= x['start'] < 3.9]
print(f"MoE raw: {len(m1)} notes in m1")
for x in m1:
    print(f"  t={x['start']:.3f} p={x['pitch']} s={x.get('string','?')} f={x.get('fret','?')}")
