import json
bd = json.load(open(r'D:\Music\nextchord-solotab\uploads\20260512-073742\beats.json','r',encoding='utf-8'))
beats = bd['beats']
intervals = [beats[i+1]-beats[i] for i in range(min(20,len(beats)-1))]
print(f'BPM={bd["bpm"]}, expected_interval={60/bd["bpm"]:.4f}s')
print(f'Actual intervals: {[round(x,3) for x in intervals]}')
print(f'Avg={sum(intervals)/len(intervals):.4f}s = {60/(sum(intervals)/len(intervals)):.1f} BPM')

# Notes per beat
notes = json.load(open(r'D:\Music\nextchord-solotab\uploads\20260512-073742\notes_assigned.json','r',encoding='utf-8'))
import numpy as np
beats_arr = np.array(beats)
for bi in range(min(9, len(beats)-1)):
    bt, nbt = beats[bi], beats[bi+1]
    count = sum(1 for n in notes if bt <= float(n['start']) < nbt)
    print(f'  beat[{bi}] ({bt:.3f}-{nbt:.3f}): {count} notes')
