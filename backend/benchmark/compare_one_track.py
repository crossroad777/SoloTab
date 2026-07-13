"""
1曲で GT vs SoloTab出力 を詳細比較するスクリプト。
音の間隔(IOI)と duration を直接対比する。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "music-transcription", "python"))

import numpy as np
import librosa

# --- Settings ---
GT_PATH = r"D:\Music\Datasets\GuitarSet\annotation\00_Rock2-85-F_comp.jams"
WAV_PATH = r"D:\Music\Datasets\GuitarSet\audio_mono-mic\00_Rock2-85-F_comp_mic.wav"
BPM = 85

def load_gt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            for d in ann.get('data', []):
                s = float(d['time']); dur = float(d['duration']); p = int(round(float(d['value'])))
                notes.append({'start': s, 'end': s+dur, 'pitch': p, 'duration': dur})
    notes.sort(key=lambda x: x['start'])
    return notes

def add_ioi(notes):
    """各ノートにIOI(次のonsetまでの時間)を追加"""
    for i, n in enumerate(notes):
        ioi = None
        for j in range(i+1, len(notes)):
            if notes[j]['start'] > n['start'] + 0.01:
                ioi = notes[j]['start'] - n['start']
                break
        n['ioi'] = ioi
    return notes

def run_pipeline(wav_path):
    """SoloTabのフルパイプラインで推論"""
    from pure_moe_transcriber import transcribe_full
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    notes = transcribe_full(y, sr)
    for n in notes:
        n['start'] = float(n['start'])
        n['end'] = float(n.get('end', n['start'] + 0.5))
        n['duration'] = n['end'] - n['start']
    notes.sort(key=lambda x: x['start'])
    return notes

def run_quantized(pred_notes, bpm):
    """music_quantizer経由で量子化したdurationを取得"""
    sec_per_beat = 60.0 / bpm
    # 正解のビートグリッド生成
    max_t = max(n['end'] for n in pred_notes) + 2
    beats = [i * sec_per_beat for i in range(int(max_t / sec_per_beat) + 4)]
    
    try:
        from music_quantizer import quantize_notes_music21
        entries = quantize_notes_music21(
            pred_notes, beats, bpm,
            time_signature="4/4", beats_per_bar=4,
            rhythm_subdivision="straight",
        )
        return entries
    except Exception as e:
        print(f"music21 quantizer failed: {e}")
        from tab_renderer import _assign_to_bars
        return _assign_to_bars(pred_notes, beats, 4)

def main():
    print(f"Track: 00_Rock2-85-F_comp (BPM={BPM})")
    print(f"sec_per_beat = {60/BPM:.3f}s, 8th = {30/BPM:.3f}s, 16th = {15/BPM:.3f}s")
    print()
    
    # 1. Load GT
    gt = add_ioi(load_gt(GT_PATH))
    print(f"=== GT: {len(gt)} notes (first 5s) ===")
    print(f"{'onset':>7} {'pitch':>5} {'dur':>6} {'IOI':>6} {'musical_dur':>12}")
    print("-" * 45)
    for n in gt:
        if n['start'] > 5.0: break
        ioi_str = f"{n['ioi']:.3f}" if n['ioi'] else "last"
        # GT IOI = musical duration
        ql = (n['ioi'] or n['duration']) / (60/BPM)
        if ql >= 0.9: dur_name = "quarter+"
        elif ql >= 0.45: dur_name = "8th"
        elif ql >= 0.2: dur_name = "16th"
        else: dur_name = "32nd"
        print(f"{n['start']:7.3f} {n['pitch']:5d} {n['duration']:6.3f} {ioi_str:>6} {dur_name:>12}")
    
    # 2. Run pipeline
    print()
    print("Running SoloTab inference...")
    pred = add_ioi(run_pipeline(WAV_PATH))
    print(f"\n=== Predicted: {len(pred)} notes (first 5s) ===")
    print(f"{'onset':>7} {'pitch':>5} {'CRNN_dur':>9} {'IOI':>6}")
    print("-" * 35)
    for n in pred:
        if n['start'] > 5.0: break
        ioi_str = f"{n['ioi']:.3f}" if n['ioi'] else "last"
        print(f"{n['start']:7.3f} {n['pitch']:5d} {n['duration']:9.3f} {ioi_str:>6}")
    
    # 3. Run quantizer
    print()
    print("Running music21 quantizer (IOI-first)...")
    entries = run_quantized(pred, BPM)
    
    sec_per_beat = 60.0 / BPM
    divs = 12
    print(f"\n=== Quantized output (first 20 entries) ===")
    print(f"{'bar':>3} {'pos':>4} {'dur_divs':>8} {'dur_sec':>7} {'pitch':>5} {'note_val':>10}")
    print("-" * 45)
    dur_names = {48:'whole', 36:'dot-half', 24:'half', 18:'dot-qtr', 12:'quarter', 
                 9:'dot-8th', 6:'8th', 4:'trip-8th', 3:'16th', 2:'32nd', 1:'64th'}
    for e in entries[:20]:
        dd = e['duration_divs']
        ds = dd / divs * sec_per_beat
        dn = dur_names.get(dd, f"({dd})")
        print(f"{e['bar']:3d} {e.get('beat_pos_in_bar',0):4d} {dd:8d} {ds:7.3f} {e['pitch']:5d} {dn:>10}")
    
    # 4. Summary stats
    print()
    print("=== IOI comparison (matched notes) ===")
    matched = 0
    ioi_errors = []
    for g in gt:
        if g['ioi'] is None: continue
        for p in pred:
            if abs(g['start'] - p['start']) < 0.05 and abs(g['pitch'] - p['pitch']) < 1:
                if p['ioi'] is not None:
                    err = abs(g['ioi'] - p['ioi'])
                    ioi_errors.append(err)
                    matched += 1
                break
    
    if ioi_errors:
        print(f"Matched: {matched} notes")
        print(f"IOI MAE: {np.mean(ioi_errors)*1000:.1f}ms")
        print(f"IOI median error: {np.median(ioi_errors)*1000:.1f}ms")
        w10 = sum(1 for e in ioi_errors if e < 0.01) / len(ioi_errors)
        w30 = sum(1 for e in ioi_errors if e < 0.03) / len(ioi_errors)
        w50 = sum(1 for e in ioi_errors if e < 0.05) / len(ioi_errors)
        print(f"IOI within 10ms: {w10:.1%}")
        print(f"IOI within 30ms: {w30:.1%}")
        print(f"IOI within 50ms: {w50:.1%}")

if __name__ == "__main__":
    main()
