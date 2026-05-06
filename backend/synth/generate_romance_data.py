# -*- coding: utf-8 -*-
"""
generate_romance_data.py — Romance de Amor風 合成訓練データ生成
================================================================
Karplus-Strong合成エンジンを使い、Em→Amのような開放弦→押弦コード変化を
含むクラシカル3連符アルペジオパターンの訓練データを大量生成する。

目的: Conformerモデルの開放弦バイアスを解消するためのデータ拡張。

出力:
  - WAVファイル (22050Hz mono)
  - JAMSファイル (string/fret/pitchアノテーション)
"""

import os
import sys
import json
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from karplus_strong import (
    KarplusStrongSynth, STANDARD_TUNING, CHORD_FINGERINGS,
    humanize_events, apply_reverb
)

# Romance de Amor風のコード進行（開放弦→押弦の変化を含む）
ROMANCE_PROGRESSIONS = [
    # オリジナルRomance
    ["Em", "Em", "Am", "Em"],
    ["Em", "Am", "Em", "Am"],
    ["Am", "Em", "Am", "Em"],
    # Emキーのバリエーション
    ["Em", "Am", "B", "Em"],
    ["Em", "C", "Am", "Em"],
    ["Em", "Am", "D", "G"],
    # Amキーのバリエーション
    ["Am", "Dm", "E7", "Am"],
    ["Am", "E", "Am", "Dm"],
    ["Am", "G", "Am", "Em"],
    # その他（開放弦→押弦を含む）
    ["G", "Am", "Em", "Am"],
    ["C", "Am", "Em", "G"],
    ["D", "Am", "Em", "Am"],
    ["Em", "Am7", "Dm", "Am"],
    ["Em", "Am", "C", "G"],
    ["Am", "Em", "G", "Am"],
    ["Em", "Dm", "Am", "Em"],
]

# Romance de Amor風3連符アルペジオ (3/4拍子)
# beat 1: bass+melody同時 → inner(2弦) → inner(3弦)
# beat 2: melody → inner(2弦) → inner(3弦)
# beat 3: melody → inner(2弦) → [省略 or inner(3弦)]
ROMANCE_PATTERN_3_4 = {
    "time_sig": (3, 4),
    "subdivisions": 9,  # 3拍 × 3連符
    "pattern": [
        (0, [6, 1]),  # beat 1: bass + melody
        (1, [2]),     # +1/3
        (2, [3]),     # +2/3
        (3, [1]),     # beat 2: melody
        (4, [2]),
        (5, [3]),
        (6, [1]),     # beat 3: melody
        (7, [2]),
        (8, [3]),
    ]
}

# 4/4拍子バリエーション
ROMANCE_PATTERN_4_4 = {
    "time_sig": (4, 4),
    "subdivisions": 12,  # 4拍 × 3連符
    "pattern": [
        (0, [6, 1]),
        (1, [2]),
        (2, [3]),
        (3, [1]),
        (4, [2]),
        (5, [3]),
        (6, [5, 1]),  # beat 3: 交互bass + melody
        (7, [2]),
        (8, [3]),
        (9, [1]),
        (10, [2]),
        (11, [3]),
    ]
}


def chord_to_voice_map(chord_name, tuning=None):
    """コード名からmelody/inner/bass各声部のピッチ・弦・フレットを返す"""
    if tuning is None:
        tuning = STANDARD_TUNING
    
    fingering = CHORD_FINGERINGS.get(chord_name)
    if fingering is None:
        return {}
    
    voices = {}
    for string_idx, fret in enumerate(fingering):
        if fret >= 0:
            string_num = string_idx + 1  # 1=6弦, ... 6=1弦
            pitch = tuning[string_idx] + fret
            voices[string_num] = {'pitch': pitch, 'fret': fret, 'string': string_num}
    
    return voices


def generate_romance_events(
    chord_progression,
    pattern=None,
    bpm=80.0,
    measures_per_chord=2,
    melody_variation=True,
    tuning=None,
):
    """
    Romance de Amor風のアルペジオイベントを生成。
    
    melody_variation=True の場合、メロディ音をコード構成音の中から
    ビートごとに変化させる（実際の演奏を模倣）。
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    if pattern is None:
        pattern = ROMANCE_PATTERN_3_4
    
    time_sig = pattern["time_sig"]
    subdivisions = pattern["subdivisions"]
    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * time_sig[0]
    sub_duration = measure_duration / subdivisions
    
    events = []
    current_time = 0.0
    
    for chord_name in chord_progression:
        voice_map = chord_to_voice_map(chord_name, tuning)
        if not voice_map:
            current_time += measure_duration * measures_per_chord
            continue
        
        # 各弦のピッチ・フレット
        # string_num: 1=最低弦(6弦), 6=最高弦(1弦)
        # SoloTab内部表記: string 6=6弦(最低), 1=1弦(最高)
        # ここでは karplus_strong.py の規約に合わせる
        
        for measure in range(measures_per_chord):
            # メロディのバリエーション（コードトーン内でランダム）
            melody_pitches = []
            if melody_variation and 6 in voice_map:
                base_pitch = voice_map[6]['pitch']  # 1弦(最高弦)
                # コード構成音を1弦上の近傍で候補を作る
                candidates = [base_pitch]
                for delta in [2, 3, 4, 5, 7]:
                    cp = base_pitch + delta
                    if cp <= tuning[5] + 12:  # 12フレットまで
                        candidates.append(cp)
                    cp = base_pitch - delta
                    if cp >= tuning[5]:
                        candidates.append(cp)
                # 各ビートでメロディを選択
                for _ in range(time_sig[0]):
                    melody_pitches.append(np.random.choice(candidates))
            
            melody_beat_idx = 0
            
            for sub_idx, string_nums in pattern["pattern"]:
                note_time = current_time + sub_idx * sub_duration
                
                for string_num in string_nums:
                    if string_num not in voice_map:
                        continue
                    
                    info = voice_map[string_num]
                    pitch = info['pitch']
                    fret = info['fret']
                    
                    # メロディ弦(1弦=string_num 6)の場合、バリエーションを適用
                    if string_num == 1 and melody_variation and melody_pitches:
                        beat_in_measure = sub_idx * time_sig[0] // subdivisions
                        if beat_in_measure < len(melody_pitches):
                            pitch = melody_pitches[beat_in_measure]
                            fret = pitch - tuning[5]  # 1弦上のフレット
                            if fret < 0:
                                fret = info['fret']
                                pitch = info['pitch']
                    
                    velocity = np.random.uniform(0.5, 0.9)
                    if string_num in (1, 2):  # 低弦（ベース）
                        velocity = min(velocity + 0.1, 0.95)
                    
                    events.append({
                        "pitch": int(pitch),
                        "start": float(note_time),
                        "duration": float(sub_duration * 1.8),
                        "velocity": float(velocity),
                        "string": int(string_num),
                        "fret": int(fret),
                        "chord": chord_name,
                    })
            
            current_time += measure_duration
    
    return events


def events_to_jams(events, duration, sr=22050):
    """ノートイベントをJAMS互換形式に変換"""
    # SoloTabの訓練で使うフォーマット
    jams_data = {
        "annotations": [],
        "file_metadata": {
            "duration": duration,
            "sr": sr,
        }
    }
    
    # 弦ごとにグループ化
    by_string = {}
    for e in events:
        s = e['string']
        if s not in by_string:
            by_string[s] = []
        by_string[s].append(e)
    
    for string_num in sorted(by_string.keys()):
        notes = by_string[string_num]
        note_data = []
        for n in notes:
            note_data.append({
                "time": n["start"],
                "duration": n["duration"],
                "value": {
                    "midi_pitch": n["pitch"],
                    "fret": n["fret"],
                    "string": string_num,
                },
            })
        jams_data["annotations"].append({
            "namespace": "note_midi",
            "string": string_num,
            "data": note_data,
        })
    
    return jams_data


def generate_dataset(
    output_dir,
    num_tracks=500,
    sr=22050,
    seed=42,
):
    """
    合成データセットを生成する。
    
    Parameters
    ----------
    output_dir : str
        出力ディレクトリ
    num_tracks : int
        生成するトラック数
    sr : int
        サンプリングレート
    seed : int
        乱数シード
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    annot_dir = os.path.join(output_dir, "annotations")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)
    
    synth = KarplusStrongSynth(sr=sr)
    manifest = []
    
    print(f"Generating {num_tracks} Romance-style tracks...")
    
    for i in range(num_tracks):
        # ランダムパラメータ
        prog_idx = i % len(ROMANCE_PROGRESSIONS)
        progression = ROMANCE_PROGRESSIONS[prog_idx]
        bpm = np.random.uniform(60, 120)
        measures_per_chord = np.random.choice([1, 2, 3])
        
        # パターン選択
        if np.random.random() < 0.7:
            pattern = ROMANCE_PATTERN_3_4
        else:
            pattern = ROMANCE_PATTERN_4_4
        
        # メロディバリエーション
        melody_var = np.random.random() < 0.6
        
        # イベント生成
        events = generate_romance_events(
            progression, pattern, bpm, measures_per_chord,
            melody_variation=melody_var,
        )
        
        if not events:
            continue
        
        # ヒューマナイズ
        events = humanize_events(events, timing_jitter=0.08)
        
        # 合成
        audio = synth.synthesize_sequence(events, randomize_params=True)
        
        # リバーブ
        if np.random.random() < 0.7:
            audio = apply_reverb(
                audio, sr=sr,
                decay=np.random.uniform(0.1, 0.4),
                delay_ms=np.random.uniform(15, 50),
            )
        
        # ノイズ追加（微量）
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.001, 0.01)
            audio += np.random.randn(len(audio)).astype(np.float32) * noise_level
        
        # 保存
        track_id = f"romance_synth_{i:04d}"
        wav_path = os.path.join(audio_dir, f"{track_id}.wav")
        jams_path = os.path.join(annot_dir, f"{track_id}.json")
        
        sf.write(wav_path, audio, sr)
        
        duration = len(audio) / sr
        jams_data = events_to_jams(events, duration, sr)
        with open(jams_path, 'w', encoding='utf-8') as f:
            json.dump(jams_data, f, indent=2)
        
        manifest.append({
            "track_id": track_id,
            "audio_path": f"audio/{track_id}.wav",
            "annotation_path": f"annotations/{track_id}.json",
            "duration": duration,
            "bpm": bpm,
            "progression": progression,
            "pattern": "3/4" if pattern == ROMANCE_PATTERN_3_4 else "4/4",
            "num_notes": len(events),
        })
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_tracks} generated")
    
    # マニフェスト保存
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    # 統計
    total_duration = sum(m['duration'] for m in manifest)
    total_notes = sum(m['num_notes'] for m in manifest)
    print(f"\nDataset generated:")
    print(f"  Tracks: {len(manifest)}")
    print(f"  Total duration: {total_duration/60:.1f} min")
    print(f"  Total notes: {total_notes}")
    print(f"  Output: {output_dir}")
    
    return manifest


if __name__ == "__main__":
    output_dir = r"D:\Music\nextchord-solotab\datasets\Romance_Synth"
    generate_dataset(output_dir, num_tracks=500)
