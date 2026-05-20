"""MoE出力(notes.json)の生データを直接分析"""
import sys, json
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

# MoE生出力を読む (notes_assigned_original or notes.json)
from pathlib import Path
sdir = Path(r'D:\Music\nextchord-solotab\uploads\20260516-052732')
# notes.json = MoE生出力
npath = sdir / 'notes.json'
ndata = json.load(open(npath, 'r', encoding='utf-8'))
notes = ndata if isinstance(ndata, list) else ndata.get('notes', ndata)
print(f'MoE raw output: {len(notes)} notes')

# beats
bdata = json.load(open(sdir / 'beats.json', 'r', encoding='utf-8'))
beats = bdata if isinstance(bdata, list) else bdata.get('beats', [])
print(f'Beats: {len(beats)}')

import numpy as np
beats_arr = np.array(beats)

MIDI = {40:'E2', 43:'G2', 45:'A2', 47:'B2', 48:'C3', 50:'D3', 52:'E3',
        55:'G3', 57:'A3', 59:'B3', 60:'C4', 62:'D4', 64:'E4', 65:'F4',
        66:'F#4', 67:'G4', 69:'A4', 71:'B4', 72:'C5', 74:'D5', 76:'E5'}

# 最初の12拍(4小節)を詳細分析
print('\n=== First 12 beats (4 bars × 3 beats) ===')
for bi in range(min(12, len(beats)-1)):
    bt = beats[bi]
    bt_end = beats[bi+1]
    bn = [n for n in notes if bt <= n['start'] < bt_end]
    pitches = [n['pitch'] for n in bn]
    pitch_names = [MIDI.get(p, f'?{p}') for p in pitches]
    print(f'  beat {bi}: t={bt:.3f}-{bt_end:.3f}s  {len(bn)} notes  {pitch_names}')
    for n in bn:
        p = n['pitch']
        s = n.get('string', '?')
        f = n.get('fret', '?')
        v = n.get('velocity', 0)
        prob = n.get('prob', n.get('confidence', '?'))
        print(f'    pitch={p}({MIDI.get(p,"?"):4s}) S{s}:F{f} vel={v} prob={prob} t={n["start"]:.3f}')

# 正解パターン：各拍は3音、beat1はbass+melody+accomp
# beat1,4,7,10 = bass beat (bar start)
# 3連符: melody, accomp, melody (又は accomp, melody, accomp)
print('\n=== Expected pattern ===')
print('M1 beat0: Bass=E2, mel=B4, acc=B3 → [E2,B4,B3] or [E2+B4, B3, B4]')
print('M1 beat1: [B3, B4, B3]')
print('M1 beat2: [B4, B3, B4]')
