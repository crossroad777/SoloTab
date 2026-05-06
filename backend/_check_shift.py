import json,sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
with open('D:/Music/nextchord-solotab/uploads/20260429-113923/notes_assigned.json') as f:
    notes=json.load(f)
    notes=notes if isinstance(notes,list) else notes.get('notes',[])

print("=== E major section 1st string notes (34-42s) ===")
for n in sorted(notes,key=lambda x:x['start']):
    if 34<=n['start']<42 and n.get('string')==1:
        shift=n.get('_modulation_pitch_shift','')
        print("t=%.2f p=%d s%df%d shift=%s" % (n['start'], n['pitch'], n['string'], n['fret'], shift))

# Also check if modulation_point is 30.8, notes before 34 should also be shifted
print("\n=== Notes 30-35s, 1st string ===")
for n in sorted(notes,key=lambda x:x['start']):
    if 30<=n['start']<35 and n.get('string')==1:
        shift=n.get('_modulation_pitch_shift','')
        print("t=%.2f p=%d s%df%d shift=%s" % (n['start'], n['pitch'], n['string'], n['fret'], shift))
