"""クラシックギター練習曲の指番号データ抽出"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import guitarpro

GP5_DIR = r"D:\Music\nextchord-solotab\datasets\gp-classical-guitar"

files = []
for root, dirs, fnames in os.walk(os.path.join(GP5_DIR)):
    for f in fnames:
        if f.endswith('.gp5') and 'dada' not in root:
            files.append(os.path.join(root, f))

print(f"Found {len(files)} GP5 files (excluding dada)")

total_notes = 0
total_lh = 0
total_rh = 0
lh_dist = {}
rh_dist = {}
examples = []

for fp in sorted(files):
    try:
        song = guitarpro.parse(fp)
    except:
        continue
    
    fname = os.path.basename(fp)
    file_lh = 0
    file_rh = 0
    file_notes = 0
    
    for track in song.tracks:
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        file_notes += 1
                        lh = getattr(note.effect, 'leftHandFinger', None)
                        if lh is not None and lh.value >= 0:
                            file_lh += 1
                            k = lh.name
                            lh_dist[k] = lh_dist.get(k, 0) + 1
                        rh = getattr(note.effect, 'rightHandFinger', None)
                        if rh is not None and rh.value >= 0:
                            file_rh += 1
                            k = rh.name
                            rh_dist[k] = rh_dist.get(k, 0) + 1
    
    total_notes += file_notes
    total_lh += file_lh
    total_rh += file_rh
    
    lh_pct = f"{file_lh/file_notes*100:.0f}%" if file_notes else "0%"
    rh_pct = f"{file_rh/file_notes*100:.0f}%" if file_notes else "0%"
    print(f"  {fname:40s} notes={file_notes:5d} LH={file_lh:4d}({lh_pct:>4s}) RH={file_rh:4d}({rh_pct:>4s})")

print(f"\n{'='*60}")
print(f"  合計: {total_notes:,} notes, LH={total_lh:,}, RH={total_rh:,}")
if lh_dist:
    print(f"  左手: {lh_dist}")
if rh_dist:
    print(f"  右手: {rh_dist}")
