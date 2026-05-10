"""
全90 soloトラック + 全360トラック ベンチマーク
CNN-first弦分類器統合版
"""
import sys, copy, os, glob
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import string_assigner
string_assigner._STRING_CLASSIFIER = None
string_assigner._STRING_CLASSIFIER_CQT_CACHE = {}

import jams
from string_assigner import assign_strings_dp, STANDARD_TUNING

ANN_DIR = r'D:\Music\Datasets\GuitarSet\annotation'
MIC_DIR = r'D:\Music\Datasets\GuitarSet\audio_mono-mic'


def benchmark(track_filter="solo"):
    """track_filter: 'solo', 'comp', or 'all'"""
    if track_filter == "all":
        jams_files = sorted(glob.glob(os.path.join(ANN_DIR, '*.jams')))
    else:
        jams_files = sorted(glob.glob(os.path.join(ANN_DIR, f'*_{track_filter}.jams')))
    
    print(f'=== {track_filter} トラック: {len(jams_files)}曲 ===')
    
    total = 0
    correct = 0
    player_stats = {}
    
    for jams_path in jams_files:
        basename = os.path.basename(jams_path).replace('.jams', '')
        player = basename[:2]
        mic_path = os.path.join(MIC_DIR, basename + '_mic.wav')
        if not os.path.exists(mic_path):
            continue
        
        jam = jams.load(jams_path)
        gt_notes = []
        note_midi_idx = 0
        for ann in jam.annotations:
            if ann.namespace != 'note_midi':
                continue
            string_num = 6 - note_midi_idx
            note_midi_idx += 1
            if string_num < 1 or string_num > 6:
                continue
            string_idx = 6 - string_num
            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                gt_fret = midi_pitch - STANDARD_TUNING[string_idx]
                if gt_fret < 0 or gt_fret > 19:
                    continue
                gt_notes.append({
                    'pitch': midi_pitch,
                    'start': float(obs.time),
                    'duration': float(obs.duration),
                    'gt_string': string_num,
                    'gt_fret': gt_fret,
                })
        
        gt_notes.sort(key=lambda n: (n['start'], n['pitch']))
        notes_input = [{'pitch': n['pitch'], 'start': n['start'], 'duration': n['duration']} for n in gt_notes]
        assigned = assign_strings_dp(copy.deepcopy(notes_input), tuning=STANDARD_TUNING, audio_path=mic_path)
        
        tc = 0
        tt = 0
        for pred, gt in zip(assigned, gt_notes):
            tt += 1
            if pred.get('string') == gt['gt_string'] and pred.get('fret') == gt['gt_fret']:
                tc += 1
        
        total += tt
        correct += tc
        
        if player not in player_stats:
            player_stats[player] = {'correct': 0, 'total': 0}
        player_stats[player]['correct'] += tc
        player_stats[player]['total'] += tt
    
    acc = correct / total if total > 0 else 0
    print(f'弦+フレット一致率: {correct}/{total} = {acc:.4f} ({acc*100:.2f}%)')
    print()
    print('プレイヤー別:')
    for p in sorted(player_stats.keys()):
        s = player_stats[p]
        pacc = s['correct'] / s['total'] if s['total'] > 0 else 0
        print(f'  Player {p}: {pacc:.4f} ({pacc*100:.2f}%) [{s["correct"]}/{s["total"]}]')
    print()
    return acc


if __name__ == "__main__":
    solo_acc = benchmark("solo")
    comp_acc = benchmark("comp")
    
    print("=" * 60)
    print(f"Solo: {solo_acc*100:.2f}%")
    print(f"Comp: {comp_acc*100:.2f}%")
