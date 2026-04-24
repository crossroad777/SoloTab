"""
karplus_strong.py — 拡張Karplus-Strongアコースティックギター合成エンジン
============================================================================
Murgul & Heizmann (2025) "Exploring Procedural Data Generation for
Automatic Acoustic Guitar Fingerpicking Transcription" に基づく実装。

4段階パイプライン:
  Stage 1: 音声合成 (このファイル)
  Stage 2: TAB/コード進行生成
  Stage 3: MIDI化 + Humanize
  Stage 4: オーディオ拡張

拡張Karplus-Strongアルゴリズム:
  遅延線 + 6つのデジタルフィルタで弦振動を物理シミュレーション。
  - Hp(z): ピック方向フィルタ
  - Hβ(z): ピック位置フィルタ（ブリッジ寄り vs サウンドホール上）
  - Hd(z): 減衰フィルタ（エネルギー減衰モデル）
  - Hs(z): スティフネス分散フィルタ（弦の硬さ）
  - Hρ(z): 端数遅延フィルタ（微細ピッチ調整）
  - HL(z): ダイナミックレベル依存ブライトネスフィルタ
"""

import numpy as np
from typing import Optional, Dict, Tuple


# ギター標準チューニング (MIDI)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4


class KarplusStrongSynth:
    """拡張Karplus-Strongアコースティックギター合成エンジン"""
    
    def __init__(self, sr: int = 22050):
        """
        Parameters
        ----------
        sr : int
            サンプリングレート (Hz)。GuitarSet互換で22050Hz。
        """
        self.sr = sr
    
    def synthesize_note(
        self,
        midi_pitch: int,
        duration: float,
        velocity: float = 0.8,
        pick_direction: float = 0.5,
        pick_position: float = 0.13,
        damping: float = 0.998,
        brightness: float = 4000.0,
        stiffness: float = 0.0,
        detune_cents: float = 0.0,
    ) -> np.ndarray:
        """
        1ノートを合成する（ウェーブテーブル方式KS + マルチモードボディ共鳴）。
        """
        from scipy.signal import butter, sosfilt, sosfiltfilt
        
        # ピッチ→周波数
        freq = 440.0 * (2.0 ** ((midi_pitch - 69 + detune_cents / 100.0) / 12.0))
        
        # ウェーブテーブル長
        N = int(round(self.sr / freq))
        if N < 2:
            N = 2
        
        num_samples = int(duration * self.sr)
        output = np.zeros(num_samples, dtype=np.float64)
        
        # === 励振信号: ローパス済みノイズ（弦をはじく音をシミュレート）===
        wavetable = np.random.randn(N)
        
        # 励振信号をローパスフィルタで温かくする（アコギの指弾き）
        # 指で弾くと高域が少なく温かい音になる
        lp_cutoff = min(4000.0, self.sr * 0.4)
        sos_excite = butter(2, lp_cutoff, btype='lowpass', fs=self.sr, output='sos')
        # sosfiltfiltは短すぎるバッファで問題が出るのでtryで
        if N > 12:
            wavetable = sosfilt(sos_excite, wavetable).astype(np.float64)
        
        # ピック位置フィルタ: コムフィルタ
        beta = max(1, int(pick_position * N))
        if beta < N:
            filtered = np.copy(wavetable)
            for i in range(beta, N):
                filtered[i] = wavetable[i] - 0.4 * wavetable[i - beta]
            wavetable = filtered
        
        # ピック方向フィルタ
        if pick_direction > 0.01:
            p = min(pick_direction * 0.4, 0.8)
            for i in range(1, N):
                wavetable[i] = (1 - p) * wavetable[i] + p * wavetable[i - 1]
        
        # 振幅をvelocityに合わせる
        peak = np.max(np.abs(wavetable))
        if peak > 0:
            wavetable *= velocity / peak
        
        # === ループフィルタ係数 ===
        # dampingを高めに設定（0.998）でサステインを長く
        # ブライトネスフィルタ係数
        wc = 2.0 * np.pi * brightness / self.sr
        cos_w = np.cos(wc)
        val = (2.0 - cos_w)
        b_coeff = val - np.sqrt(val * val - 1.0)
        b_coeff = np.clip(b_coeff, 0.0, 0.95)
        
        # 端数遅延オールパス
        exact_delay = self.sr / freq
        frac = exact_delay - N
        if frac < 0:
            frac = 0
        eta = (1.0 - frac) / (1.0 + frac)
        
        # === メインループ: ウェーブテーブル巡回 ===
        pos = 0
        prev_filtered = 0.0
        apx1 = 0.0
        apy1 = 0.0
        
        for i in range(num_samples):
            output[i] = wavetable[pos]
            
            next_pos = (pos + 1) % N
            avg = 0.5 * (wavetable[pos] + wavetable[next_pos])
            avg *= damping
            
            filtered = (1.0 - b_coeff) * avg + b_coeff * prev_filtered
            prev_filtered = filtered
            
            ap_out = eta * filtered + apx1 - eta * apy1
            apx1 = filtered
            apy1 = ap_out
            
            wavetable[pos] = ap_out
            pos = next_pos
        
        # === アコースティックギターボディ共鳴（マルチモード）===
        # 実測に基づく4つの主要共鳴モード:
        #   1. Helmholtz (空気共鳴): ~98 Hz, Q=3
        #   2. Top plate (表板1次): ~207 Hz, Q=5
        #   3. Back plate (裏板): ~360 Hz, Q=4
        #   4. Coupled mode (結合): ~520 Hz, Q=3
        # + ハイシェルフ: 2kHz以上を少しブースト（弦のアタック感）
        
        body_modes = [
            (98.0,  3.0, 0.25),   # freq, Q, gain
            (207.0, 5.0, 0.35),
            (360.0, 4.0, 0.20),
            (520.0, 3.0, 0.15),
        ]
        
        body_output = np.zeros_like(output)
        
        for mode_freq, mode_q, mode_gain in body_modes:
            w0 = 2.0 * np.pi * mode_freq / self.sr
            alpha_m = np.sin(w0) / (2.0 * mode_q)
            
            # バンドパスフィルタ係数 (constant-skirt gain)
            b0 =  alpha_m
            b1 =  0.0
            b2 = -alpha_m
            a0 =  1.0 + alpha_m
            a1 = -2.0 * np.cos(w0)
            a2 =  1.0 - alpha_m
            
            # SOS形式に変換
            sos_mode = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
            mode_out = sosfilt(sos_mode, output)
            body_output += mode_out * mode_gain
        
        # 原音とボディ共鳴をミックス
        # 原音60% + ボディ共鳴40% → アコギらしい温かさ
        output = output * 0.55 + body_output * 0.45
        
        # ハイシェルフ: アタック時の弦のブライトネスを少し残す
        sos_high = butter(1, 2000, btype='highpass', fs=self.sr, output='sos')
        high_content = sosfilt(sos_high, output)
        output += high_content * 0.15
        
        # === エンベロープ ===
        # アタック: 短いフェードイン（クリック防止）
        attack_len = min(int(0.002 * self.sr), num_samples)
        if attack_len > 0:
            output[:attack_len] *= np.linspace(0, 1, attack_len)
        
        # リリース: フェードアウト
        release_len = min(int(0.03 * self.sr), num_samples)
        if release_len > 0:
            output[-release_len:] *= np.linspace(1, 0, release_len)
        
        # 正規化
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * velocity * 0.85
        
        return output.astype(np.float32)
    
    def synthesize_chord(
        self,
        notes: list,
        duration: float,
        strum_delay: float = 0.015,
        **kwargs
    ) -> np.ndarray:
        """
        和音を合成する。ストラムのタイムラグを模擬。
        
        Parameters
        ----------
        notes : list[int]
            MIDIピッチのリスト
        duration : float
            持続時間 (秒)
        strum_delay : float
            弦間のストラム遅延 (秒)
        """
        total_delay = strum_delay * (len(notes) - 1)
        total_len = int((duration + total_delay) * self.sr)
        output = np.zeros(total_len, dtype=np.float32)
        
        for i, pitch in enumerate(notes):
            offset = int(i * strum_delay * self.sr)
            note_audio = self.synthesize_note(pitch, duration, **kwargs)
            end = min(offset + len(note_audio), total_len)
            output[offset:end] += note_audio[:end - offset]
        
        # クリッピング防止
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak * 1.05
        
        return output
    
    def synthesize_sequence(
        self,
        note_events: list,
        randomize_params: bool = True,
    ) -> np.ndarray:
        """
        ノートイベントのシーケンスを合成する。
        
        Parameters
        ----------
        note_events : list[dict]
            各イベント: {"pitch": int, "start": float, "duration": float, 
                         "velocity": float, "string": int}
        randomize_params : bool
            合成パラメータをランダム化するか
        
        Returns
        -------
        np.ndarray
            合成音声
        """
        if not note_events:
            return np.zeros(self.sr, dtype=np.float32)
        
        # 合計時間を計算
        max_end = max(e["start"] + e["duration"] for e in note_events)
        total_len = int((max_end + 0.5) * self.sr)  # 0.5秒余裕
        output = np.zeros(total_len, dtype=np.float32)
        
        for event in note_events:
            pitch = event["pitch"]
            start = event["start"]
            duration = event["duration"]
            velocity = event.get("velocity", 0.7)
            
            # 合成パラメータ
            params = {}
            if randomize_params:
                params = {
                    "pick_direction": np.random.uniform(0.0, 0.8),
                    "pick_position": np.random.uniform(0.05, 0.25),
                    "damping": np.random.uniform(0.994, 0.999),
                    "brightness": np.random.uniform(500, 3000),
                    "stiffness": np.random.uniform(0.0, 0.02),
                    "detune_cents": np.random.uniform(-10, 10),
                }
            
            note_audio = self.synthesize_note(
                pitch, duration, velocity, **params
            )
            
            offset = int(start * self.sr)
            end = min(offset + len(note_audio), total_len)
            if end > offset:
                output[offset:end] += note_audio[:end - offset]
        
        # クリッピング防止
        peak = np.max(np.abs(output))
        if peak > 0.95:
            output = output / peak * 0.9
        
        return output


# ===================================================================
# フィンガーピッキングパターンDB
# ===================================================================

# PIMA記法: P=親指(低弦), I=人差指, M=中指, A=薬指
# 弦番号: -1=最低弦から1番目, -2=2番目, 1=最高弦, 2=2番目...
# ここではシンプルに弦番号(1-6)で指定

FINGERPICKING_PATTERNS = {
    "travis_basic": {
        "name": "Travis Picking (基本)",
        "time_sig": (4, 4),
        "subdivisions": 8,  # 8分音符
        # (subdivision_index, string_numbers)
        "pattern": [
            (0, [6]),        # beat 1: 親指-ベース
            (1, [3]),        # &: 中指
            (2, [5]),        # beat 2: 親指-交互ベース
            (3, [2]),        # &: 薬指
            (4, [6]),        # beat 3: 親指-ベース
            (5, [3]),        # &: 中指
            (6, [4]),        # beat 4: 親指-交互ベース
            (7, [2, 1]),     # &: 薬指+小指
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
    "classical_p_i_m_a": {
        "name": "クラシカル PIMA",
        "time_sig": (4, 4),
        "subdivisions": 16,  # 16分音符
        "pattern": [
            (0, [6]),        # beat 1
            (1, [3]),
            (2, [2]),
            (3, [1]),
            (4, [5]),        # beat 2
            (5, [3]),
            (6, [2]),
            (7, [1]),
            (8, [4]),        # beat 3
            (9, [3]),
            (10, [2]),
            (11, [1]),
            (12, [5]),       # beat 4
            (13, [3]),
            (14, [2]),
            (15, [1]),
        ]
    },
    "fingerstyle_ballad": {
        "name": "フィンガースタイル バラード",
        "time_sig": (4, 4),
        "subdivisions": 8,
        "pattern": [
            (0, [6, 3]),     # beat 1: ベース+内声同時
            (1, [2]),
            (2, [1]),
            (3, [2]),
            (4, [5, 3]),     # beat 3: 交互ベース+内声
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
            (0, [6]),        # beat 1: ベース
            (1, [3, 2, 1]),  # &: 和音
            (2, [3, 2, 1]),  # beat 2: 和音
            (3, [5]),        # &: 交互ベース
            (4, [3, 2, 1]),  # beat 3: 和音
            (5, [3, 2, 1]),  # &: 和音
        ]
    },
}


# ===================================================================
# コード進行DB
# ===================================================================

# コードフィンガリング: {コード名: [(弦1fret, 弦2fret, ..., 弦6fret)]}
# -1 = ミュート(弾かない)
CHORD_FINGERINGS = {
    # メジャー
    "C":  [-1, 3, 2, 0, 1, 0],
    "D":  [-1, -1, 0, 2, 3, 2],
    "E":  [0, 2, 2, 1, 0, 0],
    "F":  [1, 3, 3, 2, 1, 1],
    "G":  [3, 2, 0, 0, 0, 3],
    "A":  [-1, 0, 2, 2, 2, 0],
    "B":  [-1, 2, 4, 4, 4, 2],
    
    # マイナー
    "Am": [-1, 0, 2, 2, 1, 0],
    "Dm": [-1, -1, 0, 2, 3, 1],
    "Em": [0, 2, 2, 0, 0, 0],
    "Fm": [1, 3, 3, 1, 1, 1],
    "Bm": [-1, 2, 4, 4, 3, 2],
    
    # セブンス
    "G7":  [3, 2, 0, 0, 0, 1],
    "C7":  [-1, 3, 2, 3, 1, 0],
    "D7":  [-1, -1, 0, 2, 1, 2],
    "E7":  [0, 2, 0, 1, 0, 0],
    "A7":  [-1, 0, 2, 0, 2, 0],
    "Am7": [-1, 0, 2, 0, 1, 0],
    "Dm7": [-1, -1, 0, 2, 1, 1],
    "Em7": [0, 2, 0, 0, 0, 0],
    
    # サス、アド
    "Dsus2": [-1, -1, 0, 2, 3, 0],
    "Dsus4": [-1, -1, 0, 2, 3, 3],
    "Asus2": [-1, 0, 2, 2, 0, 0],
    "Asus4": [-1, 0, 2, 2, 3, 0],
    "Cadd9": [-1, 3, 2, 0, 3, 0],
}

# コード進行パターン (キー=Cメジャー/Amマイナー基準)
CHORD_PROGRESSIONS = [
    # ポップス基本
    ["C", "G", "Am", "F"],
    ["G", "Em", "C", "D"],
    ["Am", "F", "C", "G"],
    ["C", "Am", "Dm", "G7"],
    
    # バラード
    ["C", "Em", "Am", "G"],
    ["Am", "Dm", "G", "C"],
    ["Em", "Am", "D", "G"],
    ["C", "Cadd9", "Am7", "F"],
    
    # フォーク
    ["G", "C", "D", "G"],
    ["D", "A", "Bm", "G"],
    ["A", "E", "D", "A"],
    ["Em", "C", "G", "D"],
    
    # ジャズ風
    ["Dm7", "G7", "C", "Am7"],
    ["Am7", "D7", "G", "C7"],
    ["Em7", "A7", "Dm7", "G7"],
    
    # マイナー
    ["Am", "Dm", "E7", "Am"],
    ["Dm", "Am", "E7", "Am"],
    ["Em", "Am", "D", "G"],
    ["Am", "G", "F", "E7"],
    
    # サス系
    ["Dsus2", "D", "Asus2", "A"],
    ["G", "Dsus4", "Em", "C"],
]


def chord_to_pitches(chord_name: str, tuning: list = None) -> list:
    """コード名からMIDIピッチのリストを返す"""
    if tuning is None:
        tuning = STANDARD_TUNING
    
    fingering = CHORD_FINGERINGS.get(chord_name)
    if fingering is None:
        return []
    
    pitches = []
    for string_idx, fret in enumerate(fingering):
        if fret >= 0:
            pitch = tuning[string_idx] + fret
            pitches.append((string_idx + 1, pitch, fret))
    
    return pitches


def generate_fingerpicking_events(
    chord_progression: list,
    pattern_name: str = "travis_basic",
    bpm: float = 100.0,
    measures_per_chord: int = 2,
    tuning: list = None,
) -> list:
    """
    コード進行 + フィンガーピッキングパターン → ノートイベント列を生成。
    
    Parameters
    ----------
    chord_progression : list[str]
        コード名のリスト
    pattern_name : str
        パターン名
    bpm : float
        テンポ
    measures_per_chord : int
        1コードあたりの小節数
    
    Returns
    -------
    list[dict]
        ノートイベント列 {"pitch", "start", "duration", "velocity", "string", "fret"}
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    pattern = FINGERPICKING_PATTERNS.get(pattern_name)
    if pattern is None:
        pattern = FINGERPICKING_PATTERNS["travis_basic"]
    
    time_sig = pattern["time_sig"]
    subdivisions = pattern["subdivisions"]
    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * time_sig[0]
    subdivision_duration = measure_duration / subdivisions
    
    events = []
    current_time = 0.0
    
    for chord_name in chord_progression:
        chord_pitches = chord_to_pitches(chord_name, tuning)
        if not chord_pitches:
            current_time += measure_duration * measures_per_chord
            continue
        
        # 弦番号→ピッチのマッピング
        string_pitch_map = {}
        for string_num, pitch, fret in chord_pitches:
            string_pitch_map[string_num] = (pitch, fret)
        
        for measure in range(measures_per_chord):
            for sub_idx, strings in pattern["pattern"]:
                note_time = current_time + sub_idx * subdivision_duration
                note_duration = subdivision_duration * 1.5  # レガート
                
                for string_num in strings:
                    if string_num in string_pitch_map:
                        pitch, fret = string_pitch_map[string_num]
                        velocity = np.random.uniform(0.5, 0.9)
                        
                        # ベース弦は少し強く
                        if string_num >= 4:
                            velocity = min(velocity + 0.1, 0.95)
                        
                        events.append({
                            "pitch": pitch,
                            "start": note_time,
                            "duration": note_duration,
                            "velocity": velocity,
                            "string": string_num,
                            "fret": fret,
                            "chord": chord_name,
                        })
            
            current_time += measure_duration
    
    return events


def humanize_events(events: list, timing_jitter: float = 0.1) -> list:
    """
    ノートイベントにヒューマナイズ（タイミングのゆらぎ）を適用。
    
    Parameters
    ----------
    events : list[dict]
        ノートイベント列
    timing_jitter : float
        最大タイミング偏差（ノート長の何%）
    """
    humanized = []
    for event in events:
        e = dict(event)
        dur = e["duration"]
        
        # タイミングジッタ (±10%のノート長)
        onset_jitter = np.random.uniform(-timing_jitter, timing_jitter) * dur
        e["start"] = max(0, e["start"] + onset_jitter)
        
        # 長さのジッタ (±5%)
        dur_jitter = np.random.uniform(-0.05, 0.05) * dur
        e["duration"] = max(0.05, dur + dur_jitter)
        
        # ベロシティのジッタ (±10%)
        vel_jitter = np.random.uniform(-0.1, 0.1)
        e["velocity"] = np.clip(e["velocity"] + vel_jitter, 0.3, 1.0)
        
        humanized.append(e)
    
    return humanized


def apply_reverb(audio: np.ndarray, sr: int = 22050, 
                 decay: float = 0.3, delay_ms: float = 30) -> np.ndarray:
    """シンプルなリバーブ効果を追加"""
    delay_samples = int(delay_ms * sr / 1000)
    output = np.copy(audio)
    
    # 複数のディレイラインでリバーブをシミュレート
    delays = [delay_samples, int(delay_samples * 1.5), 
              int(delay_samples * 2.3), int(delay_samples * 3.1)]
    decays = [decay, decay * 0.7, decay * 0.5, decay * 0.3]
    
    for d, dec in zip(delays, decays):
        if d < len(output):
            output[d:] += audio[:-d] * dec if d > 0 else audio * dec
    
    # 正規化
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9
    
    return output.astype(np.float32)


def generate_training_sample(
    progression_idx: int = None,
    pattern_name: str = None,
    bpm: float = None,
    sr: int = 22050,
) -> Tuple[np.ndarray, list]:
    """
    1つの完全な学習サンプル（音声 + ノートアノテーション）を生成。
    
    Returns
    -------
    (audio, events) : (np.ndarray, list[dict])
    """
    # ランダム選択
    if progression_idx is None:
        progression_idx = np.random.randint(len(CHORD_PROGRESSIONS))
    if pattern_name is None:
        pattern_name = np.random.choice(list(FINGERPICKING_PATTERNS.keys()))
    if bpm is None:
        bpm = np.random.uniform(60, 140)
    
    progression = CHORD_PROGRESSIONS[progression_idx]
    
    # ノートイベント生成
    events = generate_fingerpicking_events(
        progression, pattern_name, bpm,
        measures_per_chord=np.random.randint(1, 4)
    )
    
    # ヒューマナイズ
    events = humanize_events(events)
    
    # 音声合成
    synth = KarplusStrongSynth(sr=sr)
    audio = synth.synthesize_sequence(events, randomize_params=True)
    
    # リバーブ
    if np.random.random() < 0.7:
        audio = apply_reverb(
            audio, sr=sr,
            decay=np.random.uniform(0.1, 0.5),
            delay_ms=np.random.uniform(15, 60),
        )
    
    return audio, events


# ===================================================================
# テスト
# ===================================================================

if __name__ == "__main__":
    import soundfile as sf
    import os
    
    print("=== Karplus-Strong Acoustic Guitar Synth ===")
    print(f"Patterns: {list(FINGERPICKING_PATTERNS.keys())}")
    print(f"Progressions: {len(CHORD_PROGRESSIONS)}")
    print(f"Chords: {list(CHORD_FINGERINGS.keys())}")
    
    # テスト生成
    audio, events = generate_training_sample(
        progression_idx=0,  # C-G-Am-F
        pattern_name="travis_basic",
        bpm=100.0,
    )
    
    print(f"\nGenerated: {len(audio)/22050:.1f}s, {len(events)} notes")
    print(f"Sample notes:")
    for e in events[:10]:
        print(f"  t={e['start']:.3f} pitch={e['pitch']} "
              f"s{e['string']}f{e['fret']} v={e['velocity']:.2f} "
              f"chord={e['chord']}")
    
    # 保存
    out_dir = r"D:\datasets\acoustic_guitar\synth_test"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "test_sample.wav"), audio, 22050)
    print(f"\nSaved to {out_dir}/test_sample.wav")
