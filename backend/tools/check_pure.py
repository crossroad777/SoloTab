import json
with open('pure_moe_output.json', encoding='utf-8') as f:
    data = json.load(f)
b4_notes = [n for n in data if n.get('pitch') == 71]
print(f'Total notes: {len(data)}')
print(f'Found {len(b4_notes)} B4(pitch=71) notes.')
for n in b4_notes[:15]:
    print(f"t={n.get('start', 0):.2f}-{n.get('end', 0):.2f}, string={n.get('string')}, fret={n.get('fret')}")
