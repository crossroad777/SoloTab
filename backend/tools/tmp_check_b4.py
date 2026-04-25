import json
with open('d:/Music/nextchord-solotab/uploads/20260425-110343-yt-db36fe/notes.json', encoding='utf-8') as f:
    data = json.load(f)

# data might be a dict with "notes" or a list
notes = data if isinstance(data, list) else data.get("notes", [])
b4_notes = [n for n in notes if n.get('pitch') == 71]
print(f'Found {len(b4_notes)} B4(pitch=71) notes. Details:')
for n in b4_notes[:20]:
    print(f"t={n.get('start', 0):.2f}-{n.get('end', 0):.2f}, sources={n.get('sources', [])}, confidence={n.get('confidence')}")

# Also check raw output of dynamic_moe
with open('d:/Music/nextchord-solotab/debug_raw_dynamic_moe.json', encoding='utf-8') as f:
    moe_data = json.load(f)
moe_b4 = [n for n in moe_data if n.get('pitch') == 71]
print(f'\nDynamic MoE Found {len(moe_b4)} B4(pitch=71) notes. Details:')
for n in moe_b4[:10]:
    print(f"t={n.get('start', 0):.2f}-{n.get('end', 0):.2f}")
    
with open('d:/Music/nextchord-solotab/debug_raw_basic_pitch.json', encoding='utf-8') as f:
    bp_data = json.load(f)
bp_b4 = [n for n in bp_data if n.get('pitch') == 71]
print(f'\nBasic Pitch Found {len(bp_b4)} B4(pitch=71) notes. Details:')
for n in bp_b4[:10]:
    print(f"t={n.get('start', 0):.2f}-{n.get('end', 0):.2f}")
