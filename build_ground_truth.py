"""TABデータから正解JSONを生成するスクリプト"""
import json

# Standard tuning MIDI: 1弦E4=64, 2弦B3=59, 3弦G3=55, 4弦D3=50, 5弦A2=45, 6弦E2=40
STR_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_name(m):
    return f"{NOTE_NAMES[m%12]}{m//12-1}"

def make_notes(measure_num, chord, bass_str, bass_fret,
               mel_frets, arp2_frets, arp3_frets=None):
    """1小節分のノートリストを生成"""
    notes = []
    bass_pitch = STR_MIDI[bass_str] + bass_fret
    notes.append({"beat": 1.0, "pitch": bass_pitch, "name": midi_name(bass_pitch),
                  "string": bass_str, "fret": bass_fret, "role": "bass"})
    
    for i, (beat_num, mel_f) in enumerate(zip([1, 2, 3], mel_frets)):
        beat = float(beat_num)
        # Melody on string 1
        mp = STR_MIDI[1] + mel_f
        notes.append({"beat": beat, "pitch": mp, "name": midi_name(mp),
                      "string": 1, "fret": mel_f, "role": "melody"})
        # Arpeggio on string 2
        a2f = arp2_frets[i] if isinstance(arp2_frets, list) else arp2_frets
        ap2 = STR_MIDI[2] + a2f
        notes.append({"beat": beat + 0.33, "pitch": ap2, "name": midi_name(ap2),
                      "string": 2, "fret": a2f, "role": "arpeggio"})
        # Arpeggio on string 3 (if present)
        if arp3_frets is not None:
            a3f_list = arp3_frets if isinstance(arp3_frets, list) else [arp3_frets]*3
            if i < len(a3f_list) and a3f_list[i] is not None:
                ap3 = STR_MIDI[3] + a3f_list[i]
                notes.append({"beat": beat + 0.67, "pitch": ap3, "name": midi_name(ap3),
                              "string": 3, "fret": a3f_list[i], "role": "arpeggio"})
    return notes

measures = []

# === A Section (Em) m1-m4 ===
measures.append({"measure": 1, "chord": "Em", "notes": make_notes(1, "Em", 6, 0, [7,7,7], 0, 0)})
measures.append({"measure": 2, "chord": "Em", "notes": make_notes(2, "Em", 6, 0, [7,5,3], 0, 0)})
measures.append({"measure": 3, "chord": "Em", "notes": make_notes(3, "Em", 6, 0, [3,2,0], 0, 0)})
measures.append({"measure": 4, "chord": "Em", "notes": make_notes(4, "Em", 6, 0, [0,3,7], 0, 0)})

# === m5-m6 (Em, high position) ===
measures.append({"measure": 5, "chord": "Em", "notes": make_notes(5, "Em", 6, 0, [12,12,12], 0, 0)})
measures.append({"measure": 6, "chord": "Em", "notes": make_notes(6, "Em", 6, 0, [12,10,8], 0, 0)})

# === m7-m8 (Am) ===
measures.append({"measure": 7, "chord": "Am", "notes": make_notes(7, "Am", 5, 0, [8,7,5], 5, 5)})
measures.append({"measure": 8, "chord": "Am", "notes": make_notes(8, "Am", 5, 0, [5,7,8], 5, 5)})

# === m9-m10 (B) - has 3-string arpeggio ===
measures.append({"measure": 9, "chord": "B", "notes": make_notes(9, "B", 6, 7, [7,8,7], 7, [None,8,8])})
measures.append({"measure": 10, "chord": "B", "notes": make_notes(10, "B", 6, 7, [11,8,7], 7, [None,8,8])})

# === m11-m12 (Em) ===
measures.append({"measure": 11, "chord": "Em", "notes": make_notes(11, "Em", 6, 0, [7,5,3], 0, 0)})
measures.append({"measure": 12, "chord": "Em", "notes": make_notes(12, "Em", 6, 0, [3,2,0], 0, 0)})

# === m13 (B7) - has 3-string arpeggio ===
measures.append({"measure": 13, "chord": "B7", "notes": make_notes(13, "B7", 5, 2, [2,2,2], 0, [None,2,2])})

# === B Section ===
# m17 (E major)
measures.append({"measure": 17, "chord": "E", "notes": make_notes(17, "E", 6, 0, [4,4,4], 0, [None,1,1])})
measures.append({"measure": 18, "chord": "E", "notes": make_notes(18, "E", 6, 0, [4,2,0], 0, [None,1,1])})

# m19-m20 (B7) - melody on string 2
m19_notes = [
    {"beat": 1.0, "pitch": 47, "name": "B2", "string": 5, "fret": 2, "role": "bass"},
    {"beat": 1.0, "pitch": 64, "name": "E4", "string": 1, "fret": 0, "role": "melody"},
    {"beat": 1.33, "pitch": 64, "name": "E4", "string": 2, "fret": 5, "role": "arpeggio"},
    {"beat": 1.67, "pitch": 57, "name": "A3", "string": 3, "fret": 2, "role": "arpeggio"},
    {"beat": 2.0, "pitch": 63, "name": "D#4", "string": 2, "fret": 4, "role": "melody"},
    {"beat": 2.33, "pitch": 57, "name": "A3", "string": 3, "fret": 2, "role": "arpeggio"},
    {"beat": 3.0, "pitch": 61, "name": "C#4", "string": 2, "fret": 2, "role": "melody"},
    {"beat": 3.33, "pitch": 57, "name": "A3", "string": 3, "fret": 2, "role": "arpeggio"},
]
measures.append({"measure": 19, "chord": "B7", "notes": m19_notes})

m20_notes = [
    {"beat": 1.0, "pitch": 47, "name": "B2", "string": 5, "fret": 2, "role": "bass"},
    {"beat": 1.0, "pitch": 61, "name": "C#4", "string": 2, "fret": 2, "role": "melody"},
    {"beat": 1.33, "pitch": 57, "name": "A3", "string": 3, "fret": 2, "role": "arpeggio"},
    {"beat": 2.0, "pitch": 62, "name": "D4", "string": 2, "fret": 3, "role": "melody"},
    {"beat": 2.33, "pitch": 57, "name": "A3", "string": 3, "fret": 2, "role": "arpeggio"},
    {"beat": 3.0, "pitch": 63, "name": "D#4", "string": 2, "fret": 4, "role": "melody"},
    {"beat": 3.33, "pitch": 57, "name": "A3", "string": 3, "fret": 2, "role": "arpeggio"},
]
measures.append({"measure": 20, "chord": "B7", "notes": m20_notes})

# m21-m22 (B7 high position)
measures.append({"measure": 21, "chord": "B7", "notes": make_notes(21, "B7", 6, 7, [9,9,9], 7, [None,8,8])})
measures.append({"measure": 22, "chord": "B7", "notes": make_notes(22, "B7", 6, 7, [9,11,9], 7, [None,8,8])})

# m23-24 (E)
measures.append({"measure": 23, "chord": "E", "notes": make_notes(23, "E", 6, 0, [9,7,7], 9, 9)})

# Build full ground truth
gt = {
    "_comment": "禁じられた遊び (Romance) 正解データ v3 - 全小節TABデータ付き",
    "_source": "ユーザー提供のスクロールTab譜",
    "metadata": {
        "title": "禁じられた遊び (Romance)",
        "key_A_section": "Em", "key_B_section": "E",
        "time_signature": "3/4", "bpm": 88,
        "tuning": "standard", "tuning_pitches": [40, 45, 50, 55, 59, 64],
        "capo": 0
    },
    "structure": {
        "A_section_measures": "1-13 (repeats once)",
        "B_section_measures": "17-23+",
        "pattern_per_beat": "melody → arp_str2 [→ arp_str3]",
        "notes_per_measure": "7-9 (varies by chord voicing)"
    },
    "measures_detailed": measures,
    "pitch_summary": {}
}

# Collect pitch summary
from collections import Counter
all_pitches = Counter()
for m in measures:
    for n in m["notes"]:
        all_pitches[n["pitch"]] += 1

gt["pitch_summary"] = {f"{midi_name(p)}_{p}": cnt for p, cnt in sorted(all_pitches.items())}

out_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(gt, f, indent=2, ensure_ascii=False)

print(f"Ground truth saved: {len(measures)} measures, {sum(len(m['notes']) for m in measures)} total notes")
print(f"\nPitch distribution:")
for p, cnt in sorted(all_pitches.items()):
    print(f"  {midi_name(p):>4s} (MIDI {p:2d}): {cnt}")
