import json
c = json.load(open('D:/Music/nextchord-solotab/uploads/20260428-205331-yt-8123dc/chords.json', encoding='utf-8'))
print(f"Total: {len(c)} chords")
for x in c[:10]:
    print(f"  {x['start']:.1f}-{x['end']:.1f}: {x['chord']}")
