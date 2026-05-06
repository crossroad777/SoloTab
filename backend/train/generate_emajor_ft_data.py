"""
generate_emajor_ft_data.py — E majorアルペジオ合成データ生成 (FT用)
====================================================================
Romance de AmorのE majorセクションで必要なハイポジション
アルペジオパターンの訓練データを合成し、GuitarSet FTに追加する。

MoEが検出できないC#5, G#4, D#4等のE major固有音を学習させる。

Usage:
    python generate_emajor_ft_data.py [--num 200] [--use-fluidsynth]
"""

import os, sys, io, time, argparse
import numpy as np
import torch
import librosa

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
synth_dir = os.path.join(project_root, "synth")
sys.path.insert(0, synth_dir)
sys.path.insert(0, project_root)

mt_python_dir = os.path.join(project_root, "..", "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config

# ギター標準チューニング (MIDI)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

# E major のハイポジションコードフォーム
# Romance de Amor E major section のフィンガリング:
# Position IV (fret 4-5): E major の基本ポジション
# Position VII (fret 7-9): C#m/E の展開ポジション  
# Position IX (fret 9-12): A/E のハイポジション
E_MAJOR_FORMS = {
    # ローポジション
    "E":      {"frets": [0, 2, 2, 1, 0, 0], "name": "E major open"},
    "A":      {"frets": [-1, 0, 2, 2, 2, 0], "name": "A major open"},
    "B7":     {"frets": [-1, 2, 1, 2, 0, 2], "name": "B7"},
    
    # ミドルポジション (Romance E major section M19-M20相当)
    "E_pos4": {"frets": [0, 2, 2, 4, 5, 4], "name": "E major pos4"},
    "A_pos5": {"frets": [-1, 0, 2, 2, 5, 5], "name": "A major pos5"},
    "F#m":    {"frets": [2, 4, 4, 2, 2, 2], "name": "F#m barre"},
    
    # ハイポジション (Romance E major section M17-M18, M25-M26相当)
    "E_pos9":   {"frets": [0, 2, 2, 9, 9, 9], "name": "E major pos9 melody"},
    "A_pos9":   {"frets": [-1, 0, 2, 9, 10, 9], "name": "A major pos9"},
    "B_pos7":   {"frets": [-1, 2, 4, 8, 7, 7], "name": "B major pos7"},
    "C#m_pos9": {"frets": [-1, 4, 6, 9, 9, 9], "name": "C#m pos9"},
    "E_pos12":  {"frets": [0, 2, 2, 12, 12, 12], "name": "E major pos12 melody"},
    "A_pos12":  {"frets": [-1, 0, 2, 11, 10, 9], "name": "A pos12 descend"},

    # 超ハイポジション (M26-M27相当)
    "E_melody_high": {"frets": [0, 2, 2, 12, 12, 12], "name": "E melody fret12"},
    "B_melody_high": {"frets": [-1, 2, 4, 11, 12, 11], "name": "B melody fret11-12"},
}

# E majorキーのアルペジオ進行 (Romance de Amor風)
E_MAJOR_PROGRESSIONS = [
    # Romance de Amor E major section 忠実再現
    ["E_pos9", "E_pos9", "A_pos9", "E_pos9"],
    ["E_pos12", "E_pos12", "A_pos12", "E_pos12"],
    ["E_pos9", "A_pos9", "B_pos7", "E_pos9"],
    ["E_pos9", "C#m_pos9", "A_pos9", "B_pos7"],
    
    # ハイ↔ローの移行パターン
    ["E", "E_pos9", "A", "A_pos9"],
    ["E_pos12", "A_pos12", "B_pos7", "E_pos9"],
    ["E_pos9", "E_pos9", "E_pos12", "E_pos12"],
    
    # ミドルポジション
    ["E_pos4", "A_pos5", "F#m", "B7"],
    ["E", "A_pos5", "B7", "E_pos4"],
    ["E_pos4", "F#m", "A_pos5", "E_pos4"],
    
    # 混合（ロー→ハイ遷移学習用）
    ["E", "E_pos9", "E_pos12", "E_pos9"],
    ["A", "A_pos9", "A_pos12", "A_pos9"],
    ["E", "A", "E_pos9", "A_pos9"],
    ["B7", "E_pos9", "A_pos9", "E"],
    
    # E majorスケールの高音域学習
    ["E_melody_high", "A_pos12", "B_melody_high", "E_pos12"],
    ["E_pos9", "B_melody_high", "E_pos12", "A_pos9"],
]

# Romance de Amor風3連符アルペジオ (3/4拍子)
ROMANCE_PATTERN = {
    "time_sig": (3, 4),
    "subdivisions": 9,  # 3拍 × 3連符
    "pattern": [
        (0, [6, 1]),  # beat 1: bass + melody
        (1, [2]),     # inner
        (2, [3]),     # inner
        (3, [1]),     # beat 2: melody
        (4, [2]),
        (5, [3]),
        (6, [1]),     # beat 3: melody
        (7, [2]),
        (8, [3]),
    ]
}

# メロディにスケール音のバリエーションを追加
E_MAJOR_SCALE_FRETS_STRING1 = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16]  # E major on 1st string


def generate_events(progression, bpm=80, measures_per_chord=2, melody_var=True):
    """E majorアルペジオイベントを生成"""
    pattern = ROMANCE_PATTERN
    time_sig = pattern["time_sig"]
    subdivisions = pattern["subdivisions"]
    beat_dur = 60.0 / bpm
    measure_dur = beat_dur * time_sig[0]
    sub_dur = measure_dur / subdivisions
    
    events = []
    t = 0.0
    
    for chord_key in progression:
        form = E_MAJOR_FORMS.get(chord_key)
        if not form:
            t += measure_dur * measures_per_chord
            continue
        
        frets = form["frets"]
        # 弦→ピッチマップ: string 1=1弦(最高E4), 6=6弦(最低E2)
        # internal: index 0=6弦, 5=1弦
        # pattern string_num: 1=1弦(最高), 6=6弦(最低)
        string_pitches = {}  # string_num → (pitch, fret)
        for idx in range(6):
            f = frets[idx]
            if f >= 0:
                string_num = 6 - idx  # idx 0=6弦→string_num 6, idx 5=1弦→string_num 1
                pitch = STANDARD_TUNING[idx] + f
                string_pitches[string_num] = (pitch, f)
        
        for measure in range(measures_per_chord):
            # メロディバリエーション
            melody_candidates = []
            if melody_var and 1 in string_pitches:
                base_fret = string_pitches[1][1]
                for scale_fret in E_MAJOR_SCALE_FRETS_STRING1:
                    if abs(scale_fret - base_fret) <= 3:
                        pitch = STANDARD_TUNING[5] + scale_fret
                        melody_candidates.append((pitch, scale_fret))
                if not melody_candidates:
                    melody_candidates.append(string_pitches[1])
            
            melody_beat_pitches = []
            for _ in range(time_sig[0]):
                if melody_candidates:
                    mc = melody_candidates[np.random.randint(len(melody_candidates))]
                    melody_beat_pitches.append(mc)
                elif 1 in string_pitches:
                    melody_beat_pitches.append(string_pitches[1])
            
            for sub_idx, string_nums in pattern["pattern"]:
                note_time = t + sub_idx * sub_dur
                
                for sn in string_nums:
                    if sn not in string_pitches:
                        continue
                    
                    pitch, fret = string_pitches[sn]
                    
                    # メロディ弦バリエーション
                    if sn == 1 and melody_beat_pitches:
                        beat = sub_idx * time_sig[0] // subdivisions
                        if beat < len(melody_beat_pitches):
                            pitch, fret = melody_beat_pitches[beat]
                    
                    vel = np.random.uniform(0.5, 0.9)
                    if sn >= 5:  # 低弦
                        vel = min(vel + 0.1, 0.95)
                    
                    events.append({
                        "pitch": int(pitch),
                        "start": float(note_time),
                        "duration": float(sub_dur * 1.8),
                        "velocity": float(vel),
                        "string": int(6 - sn + 1),  # internal: 1=6弦 → string_index: 0=high E
                        "fret": int(fret),
                    })
            
            t += measure_dur
    
    # ヒューマナイズ
    for e in events:
        jitter = np.random.uniform(-0.02, 0.02)
        e["start"] = max(0, e["start"] + jitter)
        e["velocity"] = float(np.clip(e["velocity"] + np.random.uniform(-0.05, 0.05), 0.3, 1.0))
    
    return events


def events_to_labels(events, sr=22050, hop=512):
    """ノートイベントを [start, end, string_idx, fret, pitch] ラベルテンソルに変換"""
    labels = []
    for e in events:
        # string mapping: internal string (1=6弦 → index 0 in model output)
        # GuitarSet convention: string 0 = highest (E4), 5 = lowest (E2)
        # Our event "string" field is already in this format
        string_idx = e["string"] - 1  # 0-indexed
        labels.append([
            e["start"],
            e["start"] + e["duration"],
            float(string_idx),
            float(e["fret"]),
            float(e["pitch"]),
        ])
    
    if not labels:
        return torch.zeros(0, 5)
    
    return torch.tensor(labels, dtype=torch.float32)


def synthesize_with_karplus(events, sr=22050):
    """Karplus-Strongで合成（FluidSynthが使えない場合のフォールバック）"""
    from karplus_strong import KarplusStrongSynth, apply_reverb
    synth = KarplusStrongSynth(sr=sr)
    audio = synth.synthesize_sequence(events, randomize_params=True)
    audio = apply_reverb(audio, sr=sr, decay=np.random.uniform(0.15, 0.35))
    return audio


def synthesize_with_fluidsynth(events, sr=22050):
    """FluidSynthで高品質合成"""
    from guitar_synth import SoundFontSynth, GM_ACOUSTIC_NYLON
    program = np.random.choice([GM_ACOUSTIC_NYLON, 25])  # nylon or steel
    synth = SoundFontSynth(sr=sr, program=program)
    audio = synth.synthesize_sequence(events)
    return audio


def generate_and_save(output_dir, num_tracks=200, sr=22050, use_fluidsynth=False):
    """E majorアルペジオデータを生成し、GuitarSet FT互換の.ptファイルとして保存"""
    
    hop = config.HOP_LENGTH  # 512
    n_bins = config.N_BINS_CQT  # 168
    bins_per_octave = config.BINS_PER_OCTAVE_CQT  # 24
    fmin = config.FMIN_CQT
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    t0 = time.time()
    generated = 0
    total_notes = 0
    
    print(f"=== E Major Arpeggio Training Data Generation ===")
    print(f"Target: {num_tracks} tracks")
    print(f"Output: {output_dir}")
    print(f"Synthesizer: {'FluidSynth' if use_fluidsynth else 'Karplus-Strong'}")
    print(f"CQT: {n_bins} bins, hop={hop}, sr={sr}")
    print()
    
    for i in range(num_tracks):
        # ランダムパラメータ
        prog_idx = i % len(E_MAJOR_PROGRESSIONS)
        progression = E_MAJOR_PROGRESSIONS[prog_idx]
        bpm = np.random.uniform(65, 110)
        measures_per_chord = np.random.choice([1, 2, 3])
        melody_var = np.random.random() < 0.7
        
        # イベント生成
        events = generate_events(progression, bpm, measures_per_chord, melody_var)
        if not events:
            continue
        
        # 合成
        try:
            if use_fluidsynth:
                audio = synthesize_with_fluidsynth(events, sr)
            else:
                audio = synthesize_with_karplus(events, sr)
        except Exception as e:
            print(f"  [{i}] Synthesis error: {e}")
            continue
        
        if len(audio) < sr:  # 1秒未満はスキップ
            continue
        
        # ノイズ追加（リアリティ向上）
        if np.random.random() < 0.5:
            noise = np.random.randn(len(audio)).astype(np.float32) * np.random.uniform(0.001, 0.008)
            audio = audio + noise
        
        # CQT特徴量
        cqt_spec = librosa.cqt(
            y=audio, sr=sr, hop_length=hop,
            fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave
        )
        log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features = torch.tensor(log_cqt, dtype=torch.float32)
        
        # ラベル
        labels = events_to_labels(events, sr, hop)
        
        # 保存
        track_id = f"emajor_synth_{i:04d}"
        feat_path = os.path.join(output_dir, f"{track_id}_features.pt")
        label_path = os.path.join(output_dir, f"{track_id}_labels.pt")
        
        torch.save(features, feat_path)
        torch.save(labels, label_path)
        
        generated += 1
        total_notes += len(events)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_tracks}] {generated} generated, "
                  f"{total_notes} notes, {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n=== Generation Complete ===")
    print(f"Generated: {generated} tracks")
    print(f"Total notes: {total_notes}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {output_dir}")
    
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=200, help="Number of tracks to generate")
    parser.add_argument("--use-fluidsynth", action="store_true", help="Use FluidSynth instead of Karplus-Strong")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output:
        output_dir = args.output
    else:
        # GuitarSet FTのtrainディレクトリに直接追加
        output_dir = os.path.join(
            mt_python_dir, "_processed_guitarset_data", "train"
        )
    
    generate_and_save(output_dir, num_tracks=args.num, use_fluidsynth=args.use_fluidsynth)
