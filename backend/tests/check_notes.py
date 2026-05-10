import json
notes = json.load(open('D:/Music/nextchord-solotab/uploads/20260428-205331-yt-8123dc/notes_assigned.json'))
first = [n for n in notes if n['start'] < 5.0]
print(f"First 5 sec: {len(first)} notes")
for n in first:
    print(f"  t={n['start']:.3f} pitch={n['pitch']} str={n['string']} fret={n['fret']}")

# What Romance should be (Em arpeggio): pitch 40(E2), 64(E4), 59(B3), 55(G3), 52(E3)...
# Open strings: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64
print("\n--- Expected for Romance opening (Em arpeggio) ---")
print("  E2(40) str6 fret0, B3(59) str2 fret0, E4(64) str1 fret0")
print("  G3(55) str3 fret0, B3(59) str2 fret0, E4(64) str1 fret0")
