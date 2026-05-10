"""Test: what does _strip_to_tab_staff actually output for AlphaTab?"""
import sys
import xml.etree.ElementTree as ET

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

with open(f"{SESSION}/tab.musicxml", "r", encoding="utf-8") as f:
    original = f.read()

# Replicate _strip_to_tab_staff from main.py
STRING_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
NOTE_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
NOTE_ALTER = [ 0,   1,   0,   1,   0,   0,   1,   0,   1,   0,   1,   0]

root = ET.fromstring(original)
staves_el = root.find(".//attributes/staves")
print(f"Original staves: {staves_el.text if staves_el is not None else 'none'}")

part = root.find(".//part[@id='P1']")
if part is None:
    print("No part P1 found!")
    sys.exit(1)

for measure in part.findall("measure"):
    attrs = measure.find("attributes")
    if attrs is not None:
        st = attrs.find("staves")
        if st is not None:
            st.text = "1"
        for clef in attrs.findall("clef"):
            if clef.get("number") == "1":
                attrs.remove(clef)
        for clef in attrs.findall("clef"):
            if clef.get("number") == "2":
                clef.set("number", "1")
        for sd in attrs.findall("staff-details"):
            if sd.get("number") == "2":
                sd.set("number", "1")
    
    to_remove = []
    for elem in measure:
        if elem.tag == "note":
            staff = elem.find("staff")
            if staff is not None and staff.text == "1":
                to_remove.append(elem)
            elif staff is not None and staff.text == "2":
                staff.text = "1"
                fret_el = elem.find(".//fret")
                str_el = elem.find(".//string")
                pitch_el = elem.find("pitch")
                if fret_el is not None and str_el is not None and pitch_el is not None:
                    s = int(str_el.text)
                    f = int(fret_el.text)
                    if s in STRING_MIDI:
                        midi = STRING_MIDI[s] + f
                        step_name = NOTE_NAMES[midi % 12]
                        alter = NOTE_ALTER[midi % 12]
                        octave = (midi // 12) - 1
                        
                        step_el = pitch_el.find("step")
                        if step_el is not None:
                            step_el.text = step_name
                        alter_el = pitch_el.find("alter")
                        if alter != 0:
                            if alter_el is None:
                                alter_el = ET.SubElement(pitch_el, "alter")
                            alter_el.text = str(alter)
                        elif alter_el is not None:
                            pitch_el.remove(alter_el)
                        octave_el = pitch_el.find("octave")
                        if octave_el is not None:
                            octave_el.text = str(octave)
        elif elem.tag == "backup":
            to_remove.append(elem)
        elif elem.tag == "forward":
            fwd_staff = elem.find("staff")
            if fwd_staff is not None and fwd_staff.text == "1":
                to_remove.append(elem)
    
    for elem in to_remove:
        measure.remove(elem)

stripped = ET.tostring(root, encoding="unicode")

# Analyze stripped output
stripped_root = ET.fromstring(stripped)
staves_after = stripped_root.find(".//attributes/staves")
print(f"Stripped staves: {staves_after.text if staves_after is not None else 'none'}")

clefs = stripped_root.findall(".//clef")
for c in clefs:
    sign = c.find("sign")
    print(f"  Clef: sign={sign.text if sign is not None else '?'} number={c.get('number','?')}")

# Check first 20 notes in stripped output
print("\nFirst 20 notes in STRIPPED output (what AlphaTab sees):")
count = 0
for note in stripped_root.iter("note"):
    rest = note.find("rest")
    if rest is not None:
        continue
    pitch = note.find("pitch")
    if pitch is None:
        continue
    step = pitch.find("step")
    octave = pitch.find("octave")
    alter = pitch.find("alter")
    fret_el = note.find(".//fret")
    str_el = note.find(".//string")
    
    step_t = step.text if step is not None else "?"
    oct_t = octave.text if octave is not None else "?"
    alt_t = "#" if (alter is not None and alter.text == "1") else ""
    fret_t = fret_el.text if fret_el is not None else "?"
    str_t = str_el.text if str_el is not None else "?"
    
    # What MIDI would AlphaTab calculate from this pitch?
    step_to_base = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
    midi_from_pitch = step_to_base.get(step_t, 0) + (int(oct_t)+1)*12
    if alter is not None:
        midi_from_pitch += int(alter.text)
    
    # What fret would AlphaTab assign? (assuming standard tuning, string from <technical>)
    s_int = int(str_t) if str_t != "?" else 1
    alphatab_fret = midi_from_pitch - STRING_MIDI.get(s_int, 64)
    
    match = "OK" if str(alphatab_fret) == fret_t else f"MISMATCH(AT={alphatab_fret})"
    
    print(f"  [{count}] str={str_t} fret={fret_t} pitch={step_t}{alt_t}{oct_t} midi={midi_from_pitch} AT_fret={alphatab_fret} {match}")
    count += 1
    if count >= 20:
        break

# Count total mismatches
total_mismatch = 0
total_notes = 0
for note in stripped_root.iter("note"):
    rest = note.find("rest")
    if rest is not None:
        continue
    pitch = note.find("pitch")
    fret_el = note.find(".//fret")
    str_el = note.find(".//string")
    if pitch is None or fret_el is None or str_el is None:
        continue
    step = pitch.find("step")
    octave = pitch.find("octave")
    alter = pitch.find("alter")
    if step is None or octave is None:
        continue
    midi_from_pitch = step_to_base.get(step.text, 0) + (int(octave.text)+1)*12
    if alter is not None:
        midi_from_pitch += int(alter.text)
    s_int = int(str_el.text)
    alphatab_fret = midi_from_pitch - STRING_MIDI.get(s_int, 64)
    total_notes += 1
    if alphatab_fret != int(fret_el.text):
        total_mismatch += 1

print(f"\nTotal notes: {total_notes}, Mismatches: {total_mismatch}")
print("If 0 mismatches, AlphaTab should display correct frets.")
