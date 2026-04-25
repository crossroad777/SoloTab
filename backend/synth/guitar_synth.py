"""
guitar_synth.py — SoundFontベースのアコースティックギター合成エンジン
====================================================================
FluidSynth + FluidR3_GM SoundFont (GM 25: Acoustic Guitar Steel) で
リアルなアコギ音の学習データを大量生成する。

パイプライン:
  1. コード進行 + フィンガーピッキングパターンからノートイベント生成
  2. ヒューマナイズ（タイミング/ベロシティのゆらぎ）
  3. FluidSynthでオーディオレンダリング
  4. オーディオ拡張（リバーブ等はFluidSynth内蔵）
"""

import numpy as np
import os
from typing import Tuple

# ギター標準チューニング (MIDI)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

# デフォルトSoundFontパス
DEFAULT_SF2 = r"D:\datasets\acoustic_guitar\FluidR3_GM.sf2"

# GM Program Numbers for guitar
GM_ACOUSTIC_STEEL = 25   # Acoustic Guitar (steel)
GM_ACOUSTIC_NYLON = 24   # Acoustic Guitar (nylon)
GM_ELECTRIC_CLEAN = 27   # Electric Guitar (clean)


class SoundFontSynth:
    """FluidSynthベースのアコギ合成エンジン"""
    
    def __init__(self, sr: int = 22050, sf2_path: str = None,
                 program: int = GM_ACOUSTIC_STEEL):
        """
        Parameters
        ----------
        sr : int
            サンプリングレート
        sf2_path : str
            SoundFontファイルパス
        program : int
            GM Program番号 (25=Steel, 24=Nylon)
        """
        import fluidsynth
        self.sr = sr
        self.sf2_path = sf2_path or DEFAULT_SF2
        self.program = program
        
        if not os.path.exists(self.sf2_path):
            raise FileNotFoundError(f"SoundFont not found: {self.sf2_path}")
        
        self._fs = fluidsynth.Synth(samplerate=float(sr))
        self._sfid = self._fs.sfload(self.sf2_path)
        if self._sfid < 0:
            raise RuntimeError(f"Failed to load SoundFont: {self.sf2_path}")
        self._fs.program_select(0, self._sfid, 0, program)
        
        # リバーブ設定（部屋の反響）
        self._fs.setting('synth.reverb.active', 1)
        self._fs.setting('synth.reverb.room-size', 0.4)
        self._fs.setting('synth.reverb.damp', 0.3)
        self._fs.setting('synth.reverb.level', 0.3)
    
    def __del__(self):
        try:
            self._fs.delete()
        except Exception:
            pass
    
    def synthesize_sequence(self, note_events: list) -> np.ndarray:
        """
        ノートイベント列をオーディオに合成する。
        
        Parameters
        ----------
        note_events : list[dict]
            各イベント: {"pitch": int, "start": float, "duration": float,
                         "velocity": float}
        
        Returns
        -------
        np.ndarray
            モノラル音声 (float32)
        """
        if not note_events:
            return np.zeros(self.sr, dtype=np.float32)
        
        # 時間でソート
        events_sorted = sorted(note_events, key=lambda e: e["start"])
        
        # 終了時間を計算
        max_end = max(e["start"] + e["duration"] for e in events_sorted)
        total_ms = int((max_end + 1.0) * 1000)  # 1秒余裕
        
        # FluidSynthを新しく初期化（前の状態をリセット）
        import fluidsynth
        fs = fluidsynth.Synth(samplerate=float(self.sr))
        sfid = fs.sfload(self.sf2_path)
        fs.program_select(0, sfid, 0, self.program)
        
        # noteoff管理
        active_notes = []  # (end_time_ms, pitch)
        
        all_chunks = []
        current_ms = 0
        
        for event in events_sorted:
            t_ms = int(event["start"] * 1000)
            pitch = event["pitch"]
            vel = int(np.clip(event.get("velocity", 0.7) * 127, 20, 127))
            dur_ms = int(event["duration"] * 1000)
            
            # この時間まで進める
            if t_ms > current_ms:
                delta = t_ms - current_ms
                n_samp = int(self.sr * delta / 1000)
                if n_samp > 0:
                    chunk = np.array(fs.get_samples(n_samp),
                                     dtype=np.float32) / 32768.0
                    all_chunks.append(chunk)
                current_ms = t_ms
            
            # 終了したノートをoff
            still_active = []
            for end_t, p in active_notes:
                if end_t <= current_ms:
                    fs.noteoff(0, p)
                else:
                    still_active.append((end_t, p))
            active_notes = still_active
            
            # ノートon
            fs.noteon(0, pitch, vel)
            active_notes.append((t_ms + dur_ms, pitch))
        
        # 残り時間を合成
        remaining = total_ms - current_ms
        if remaining > 0:
            n_samp = int(self.sr * remaining / 1000)
            if n_samp > 0:
                chunk = np.array(fs.get_samples(n_samp),
                                 dtype=np.float32) / 32768.0
                all_chunks.append(chunk)
        
        fs.delete()
        
        if not all_chunks:
            return np.zeros(self.sr, dtype=np.float32)
        
        # ステレオ→モノ変換
        audio = np.concatenate(all_chunks)
        mono = audio.reshape(-1, 2).mean(axis=1)
        
        # 正規化
        peak = np.max(np.abs(mono))
        if peak > 0.01:
            mono = mono / peak * 0.85
        
        return mono.astype(np.float32)


# ===================================================================
# フィンガーピッキングパターンDB
# ===================================================================

FINGERPICKING_PATTERNS = {
    "travis_basic": {
        "name": "Travis Picking (基本)",
        "time_sig": (4, 4),
        "subdivisions": 8,
        "pattern": [
            (0, [6]),
            (1, [3]),
            (2, [5]),
            (3, [2]),
            (4, [6]),
            (5, [3]),
            (6, [4]),
            (7, [2, 1]),
        ]
    },
    "arpeggio_up": {
        "name": "アルペジオ (上昇)",
        "time_sig": (4, 4),
        "subdivisions": 8,
        "pattern": [
            (0, [6]),
            (1, [5]),
            (2, [4]),
            (3, [3]),
            (4, [2]),
            (5, [1]),
            (6, [2]),
            (7, [3]),
        ]
    },
    "arpeggio_down": {
        "name": "アルペジオ (下降)",
        "time_sig": (4, 4),
        "subdivisions": 8,
        "pattern": [
            (0, [1]),
            (1, [2]),
            (2, [3]),
            (3, [4]),
            (4, [5]),
            (5, [6]),
            (6, [5]),
            (7, [4]),
        ]
    },
    "classical_pima": {
        "name": "クラシカル PIMA",
        "time_sig": (4, 4),
        "subdivisions": 16,
        "pattern": [
            (0, [6]), (1, [3]), (2, [2]), (3, [1]),
            (4, [5]), (5, [3]), (6, [2]), (7, [1]),
            (8, [4]), (9, [3]), (10, [2]), (11, [1]),
            (12, [5]), (13, [3]), (14, [2]), (15, [1]),
        ]
    },
    "fingerstyle_ballad": {
        "name": "フィンガースタイル バラード",
        "time_sig": (4, 4),
        "subdivisions": 8,
        "pattern": [
            (0, [6, 3]),
            (1, [2]),
            (2, [1]),
            (3, [2]),
            (4, [5, 3]),
            (5, [2]),
            (6, [1]),
            (7, [2]),
        ]
    },
    "waltz_3_4": {
        "name": "ワルツ 3/4",
        "time_sig": (3, 4),
        "subdivisions": 6,
        "pattern": [
            (0, [6]),
            (1, [3, 2, 1]),
            (2, [3, 2, 1]),
            (3, [5]),
            (4, [3, 2, 1]),
            (5, [3, 2, 1]),
        ]
    },
}


# ===================================================================
# コードDB
# ===================================================================

CHORD_FINGERINGS = {
    "C":  [-1, 3, 2, 0, 1, 0],
    "D":  [-1, -1, 0, 2, 3, 2],
    "E":  [0, 2, 2, 1, 0, 0],
    "F":  [1, 3, 3, 2, 1, 1],
    "G":  [3, 2, 0, 0, 0, 3],
    "A":  [-1, 0, 2, 2, 2, 0],
    "B":  [-1, 2, 4, 4, 4, 2],
    "Am": [-1, 0, 2, 2, 1, 0],
    "Dm": [-1, -1, 0, 2, 3, 1],
    "Em": [0, 2, 2, 0, 0, 0],
    "Fm": [1, 3, 3, 1, 1, 1],
    "Bm": [-1, 2, 4, 4, 3, 2],
    "G7":  [3, 2, 0, 0, 0, 1],
    "C7":  [-1, 3, 2, 3, 1, 0],
    "D7":  [-1, -1, 0, 2, 1, 2],
    "E7":  [0, 2, 0, 1, 0, 0],
    "A7":  [-1, 0, 2, 0, 2, 0],
    "Am7": [-1, 0, 2, 0, 1, 0],
    "Dm7": [-1, -1, 0, 2, 1, 1],
    "Em7": [0, 2, 0, 0, 0, 0],
    "Dsus2": [-1, -1, 0, 2, 3, 0],
    "Dsus4": [-1, -1, 0, 2, 3, 3],
    "Asus2": [-1, 0, 2, 2, 0, 0],
    "Asus4": [-1, 0, 2, 2, 3, 0],
    "Cadd9": [-1, 3, 2, 0, 3, 0],
}

CHORD_PROGRESSIONS = [
    ["C", "G", "Am", "F"],
    ["G", "Em", "C", "D"],
    ["Am", "F", "C", "G"],
    ["C", "Am", "Dm", "G7"],
    ["C", "Em", "Am", "G"],
    ["Am", "Dm", "G", "C"],
    ["Em", "Am", "D", "G"],
    ["C", "Cadd9", "Am7", "F"],
    ["G", "C", "D", "G"],
    ["D", "A", "Bm", "G"],
    ["A", "E", "D", "A"],
    ["Em", "C", "G", "D"],
    ["Dm7", "G7", "C", "Am7"],
    ["Am7", "D7", "G", "C7"],
    ["Em7", "A7", "Dm7", "G7"],
    ["Am", "Dm", "E7", "Am"],
    ["Dm", "Am", "E7", "Am"],
    ["Em", "Am", "D", "G"],
    ["Am", "G", "F", "E7"],
    ["Dsus2", "D", "Asus2", "A"],
    ["G", "Dsus4", "Em", "C"],
]


def chord_to_pitches(chord_name: str, tuning: list = None) -> list:
    """コード名→(弦番号, MIDIピッチ, フレット) のリスト"""
    if tuning is None:
        tuning = STANDARD_TUNING
    fingering = CHORD_FINGERINGS.get(chord_name)
    if fingering is None:
        return []
    pitches = []
    for i, fret in enumerate(fingering):
        if fret >= 0:
            pitches.append((i + 1, tuning[i] + fret, fret))
    return pitches


def generate_fingerpicking_events(
    chord_progression: list,
    pattern_name: str = "travis_basic",
    bpm: float = 100.0,
    measures_per_chord: int = 2,
    tuning: list = None,
) -> list:
    """コード進行 + パターン → ノートイベント列"""
    if tuning is None:
        tuning = STANDARD_TUNING
    
    pattern = FINGERPICKING_PATTERNS.get(pattern_name,
                FINGERPICKING_PATTERNS["travis_basic"])
    
    time_sig = pattern["time_sig"]
    subdivisions = pattern["subdivisions"]
    beat_dur = 60.0 / bpm
    measure_dur = beat_dur * time_sig[0]
    sub_dur = measure_dur / subdivisions
    
    events = []
    t = 0.0
    
    for chord_name in chord_progression:
        chord = chord_to_pitches(chord_name, tuning)
        if not chord:
            t += measure_dur * measures_per_chord
            continue
        
        smap = {s: (p, f) for s, p, f in chord}
        
        for _ in range(measures_per_chord):
            for sub_idx, strings in pattern["pattern"]:
                note_time = t + sub_idx * sub_dur
                note_dur = sub_dur * 1.8  # レガート
                
                for s in strings:
                    if s in smap:
                        pitch, fret = smap[s]
                        vel = np.random.uniform(0.5, 0.85)
                        if s >= 4:
                            vel = min(vel + 0.1, 0.95)
                        
                        events.append({
                            "pitch": pitch,
                            "start": note_time,
                            "duration": note_dur,
                            "velocity": vel,
                            "string": s,
                            "fret": fret,
                            "chord": chord_name,
                        })
            t += measure_dur
    
    return events


def humanize_events(events: list, timing_jitter: float = 0.08) -> list:
    """タイミング/ベロシティのゆらぎを追加"""
    humanized = []
    for event in events:
        e = dict(event)
        dur = e["duration"]
        e["start"] = max(0, e["start"] + np.random.uniform(-timing_jitter, timing_jitter) * dur)
        e["duration"] = max(0.05, dur + np.random.uniform(-0.05, 0.05) * dur)
        e["velocity"] = float(np.clip(e["velocity"] + np.random.uniform(-0.08, 0.08), 0.3, 1.0))
        humanized.append(e)
    return humanized


def generate_training_sample(
    progression_idx: int = None,
    pattern_name: str = None,
    bpm: float = None,
    sr: int = 22050,
    sf2_path: str = None,
    program: int = GM_ACOUSTIC_STEEL,
) -> Tuple[np.ndarray, list]:
    """
    1つの完全な学習サンプル（音声 + アノテーション）を生成。
    """
    if progression_idx is None:
        progression_idx = np.random.randint(len(CHORD_PROGRESSIONS))
    if pattern_name is None:
        pattern_name = np.random.choice(list(FINGERPICKING_PATTERNS.keys()))
    if bpm is None:
        bpm = np.random.uniform(60, 140)
    
    progression = CHORD_PROGRESSIONS[progression_idx]
    measures = np.random.randint(1, 4)
    
    events = generate_fingerpicking_events(
        progression, pattern_name, bpm, measures
    )
    events = humanize_events(events)
    
    synth = SoundFontSynth(sr=sr, sf2_path=sf2_path, program=program)
    audio = synth.synthesize_sequence(events)
    
    return audio, events


# ===================================================================
# バッチ生成
# ===================================================================

def generate_batch(
    num_samples: int = 100,
    output_dir: str = r"D:\datasets\acoustic_guitar\synth_training",
    sr: int = 22050,
    sf2_path: str = None,
) -> None:
    """
    学習用合成データをバッチ生成する。
    
    各サンプルごとに .wav + .json (アノテーション) を出力。
    """
    import json
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    t0 = time.time()
    total_duration = 0
    total_notes = 0
    
    import soundfile as sf
    
    for i in range(num_samples):
        audio, events = generate_training_sample(sr=sr, sf2_path=sf2_path)
        dur = len(audio) / sr
        total_duration += dur
        total_notes += len(events)
        
        # WAV保存
        sf.write(os.path.join(output_dir, f"synth_{i:05d}.wav"), audio, sr)
        
        # アノテーション保存
        annotation = {
            "notes": [{
                "pitch": e["pitch"],
                "start": round(e["start"], 4),
                "end": round(e["start"] + e["duration"], 4),
                "velocity": round(e["velocity"], 3),
                "string": e["string"],
                "fret": e["fret"],
            } for e in events]
        }
        with open(os.path.join(output_dir, f"synth_{i:05d}.json"), "w") as f:
            json.dump(annotation, f, indent=2)
        
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            speed = total_duration / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_samples}] "
                  f"{dur:.1f}s, {len(events)} notes "
                  f"(total: {total_duration/60:.1f}min, {speed:.1f}x RT)")
    
    elapsed = time.time() - t0
    print(f"\n=== Batch Complete ===")
    print(f"Samples: {num_samples}")
    print(f"Total duration: {total_duration/60:.1f}min ({total_duration/3600:.2f}h)")
    print(f"Total notes: {total_notes}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Speed: {total_duration/elapsed:.1f}x realtime")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    import sys
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = sys.argv[2] if len(sys.argv) > 2 else r"D:\datasets\acoustic_guitar\synth_training"
    
    print(f"=== SoundFont Guitar Synth - Batch Generation ===")
    print(f"Generating {n} samples to {out}")
    generate_batch(num_samples=n, output_dir=out)
