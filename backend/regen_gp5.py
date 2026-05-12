"""Quick re-test: simulate pipeline's theory+GP5 steps with corrected logic."""
import sys, json
import numpy as np
sys.path.insert(0, '.')

session_dir = r'D:\Music\nextchord-solotab\uploads\20260512-073742'
with open(f'{session_dir}/notes_assigned.json', 'r', encoding='utf-8') as f:
    notes = json.load(f)

# Use BPM-corrected beats
bpm = 89.1
true_interval = 60.0 / bpm
first_note = min(float(n['start']) for n in notes)
first_beat = first_note - 0.01
true_beats = [first_beat + i * true_interval for i in range(200)]
true_beats = [b for b in true_beats if b <= max(float(n['start']) for n in notes) + true_interval]

# Detect rhythm with corrected beats
from music_theory import detect_rhythm_pattern
rhythm_info = detect_rhythm_pattern(notes, true_beats)
print(f"Initial rhythm: {rhythm_info['subdivision']} (triplet_ratio={rhythm_info['triplet_ratio']:.2f})")

# Apply 3/4 arpeggio correction (same logic as pipeline.py)
if rhythm_info["subdivision"] in ("straight", "mixed"):
    beats_arr = np.array(true_beats)
    notes_per_beat = []
    for bi in range(min(len(true_beats)-1, 60)):
        bt, nbt = true_beats[bi], true_beats[bi+1]
        count = sum(1 for n in notes if bt <= float(n["start"]) < nbt)
        if count > 0:
            notes_per_beat.append(count)
    if notes_per_beat:
        avg_npb = np.mean(notes_per_beat)
        three_ratio = sum(1 for c in notes_per_beat if c == 3) / len(notes_per_beat)
        two_or_three = sum(1 for c in notes_per_beat if c in [2, 3]) / len(notes_per_beat)
        print(f"Notes/beat: avg={avg_npb:.1f}, 3-note ratio={three_ratio:.0%}, 2or3 ratio={two_or_three:.0%}")
        if avg_npb >= 2.0 and two_or_three >= 0.7:
            rhythm_info["subdivision"] = "triplet"
            rhythm_info["triplet_ratio"] = three_ratio
            print(f"-> Corrected to TRIPLET")

print(f"Final rhythm: {rhythm_info['subdivision']}")

# Generate GP5
from gp_renderer import notes_to_gp5
gp5_bytes = notes_to_gp5(
    notes, beats=true_beats, bpm=bpm, title='Romance',
    time_signature='3/4',
    rhythm_info=rhythm_info,
    noise_gate=0.20,
)
out_path = r'D:\Music\禁じられた遊び　(ロマンス ) ギター Tab譜 楽譜　コードネーム付 - アコースティック 名曲 ギター タブ 楽譜ギター タブ譜 (128k).gp5'
with open(out_path, 'wb') as f:
    f.write(gp5_bytes)
print(f'\nGP5: {len(gp5_bytes)} bytes')

# Verify
import guitarpro as gp
song = gp.parse(out_path)
t = song.tracks[0]
ok = 0
total = min(12, len(t.measures))
for mi in range(total):
    m = t.measures[mi]
    v = m.voices[0]
    nc = sum(1 for b in v.beats if 'rest' not in str(b.status).lower())
    rc = sum(1 for b in v.beats if 'rest' in str(b.status).lower())
    cc = sum(1 for b in v.beats if len(b.notes) > 1 and 'rest' not in str(b.status).lower())
    is_ok = nc >= 8 and rc == 0 and cc == 0
    if is_ok: ok += 1
    print(f"  M{mi+1:2d}: notes={nc} rests={rc} chords={cc} {'OK' if is_ok else 'NG'}")
print(f"\n  {ok}/{total} OK")
