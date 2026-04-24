"""
guitar_transcriber.py — ギター専用CRNN推論モジュール
=====================================================
学習済みGuitarTabCRNNモデルを使って、WAVファイルから
弦・フレット情報付きのノートリストを直接出力する。

Basic Pitchの2段階方式（ピッチ検出→弦割り当て）を置き換え、
1ステップで正確なTAB情報を生成する。
"""

import os
import sys
import json
import numpy as np
import torch
import librosa
from pathlib import Path

# music-transcription リポジトリのパスを追加
_MT_DIR = Path(__file__).resolve().parent.parent / "music-transcription" / "python"
if str(_MT_DIR) not in sys.path:
    sys.path.insert(0, str(_MT_DIR))

# --- 定数 ---
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS_CQT = 168
BINS_PER_OCTAVE_CQT = 24
FMIN_CQT = librosa.note_to_hz('E2')  # 82.41 Hz
MAX_FRETS = 20
MIN_NOTE_DURATION_FRAMES = 2
FRET_SILENCE_CLASS = MAX_FRETS + 1  # class index for "no note"

# 標準チューニングのオープン弦 MIDI ノート番号 (6弦→1弦)
OPEN_STRING_PITCHES = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

# --- モデルキャッシュ ---
_model_cache = {}


def _get_model_paths():
    """学習済みモデルのパスを取得する。"""
    base = _MT_DIR / "_processed_guitarset_data" / "training_output"
    target_dir = base / "baseline_model"
    
    if not target_dir.exists():
        return None, None

    model_path = target_dir / "best_model.pth"
    config_path = target_dir / "run_configuration.json"
    
    if model_path.exists() and config_path.exists():
        return str(model_path), str(config_path)

    return None, None


def _load_model(device=None):
    """学習済みCRNNモデルを読み込み、キャッシュする。"""
    global _model_cache

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = str(device)
    if cache_key in _model_cache:
        return _model_cache[cache_key], device

    model_path, config_path = _get_model_paths()
    if model_path is None:
        raise FileNotFoundError(
            "学習済みモデルが見つかりません。"
            "まず music-transcription/python/train_model.py で学習を完了してください。"
        )

    print(f"[guitar_transcriber] Loading CRNN model from: {model_path}")

    # config.json からハイパーパラメータを読み取る
    with open(config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)

    hyperparams = run_config["hyperparameters_tuned"]

    # CNN出力次元を計算（architecture用）
    from model.architecture import TabCNN, GuitarTabCRNN

    temp_cnn = TabCNN()
    with torch.no_grad():
        dummy = torch.randn(1, 1, N_BINS_CQT, 32)
        out = temp_cnn(dummy)
        cnn_out_dim = out.shape[1] * out.shape[2]
    del temp_cnn

    # モデル構築
    model = GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hyperparams.get("RNN_TYPE", "GRU"),
        rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams["RNN_LAYERS"],
        rnn_dropout=hyperparams["RNN_DROPOUT"],
        rnn_bidirectional=hyperparams.get("RNN_BIDIRECTIONAL", True),
    )

    # 重みを読み込み
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model_cache[cache_key] = model
    print(f"[guitar_transcriber] Model loaded on {device}")
    return model, device


def _extract_cqt(wav_path: str) -> torch.Tensor:
    """WAVファイルからCQTスペクトログラムを抽出する。"""
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    cqt = librosa.cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN_CQT,
        n_bins=N_BINS_CQT,
        bins_per_octave=BINS_PER_OCTAVE_CQT,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # [1, N_BINS_CQT, T] の3D形状に
    cqt_multi = np.expand_dims(log_cqt, axis=0)
    return torch.tensor(cqt_multi, dtype=torch.float32)


def _frames_to_notes(onset_binary, fret_indices, tuning_pitches=None,
                     onset_threshold=0.7):
    """
    フレーム単位の予測結果をノートリストに変換する。
    オンセット確率のピークピッキングを行い、確実な立ち上がりエッジのみを発音とする。
    長さは次の発音まで（最大1.5秒）を維持する。
    """
    if tuning_pitches is None:
        tuning_pitches = OPEN_STRING_PITCHES

    time_per_frame = HOP_LENGTH / SAMPLE_RATE
    num_frames, num_strings = onset_binary.shape
    notes = []

    for string_idx in range(num_strings):
        onsets = []
        for frame_idx in range(1, num_frames - 1):
            prob = onset_binary[frame_idx, string_idx]
            if prob > onset_threshold:
                # 局所的マキシマム（ピーク）
                if prob > onset_binary[frame_idx - 1, string_idx] and prob >= onset_binary[frame_idx + 1, string_idx]:
                    onsets.append(frame_idx)
        
        # エッジケース
        if num_frames > 1 and onset_binary[0, string_idx] > onset_threshold and onset_binary[0, string_idx] >= onset_binary[1, string_idx]:
            onsets.insert(0, 0)
        if num_frames > 1 and onset_binary[-1, string_idx] > onset_threshold and onset_binary[-1, string_idx] > onset_binary[-2, string_idx]:
            onsets.append(num_frames - 1)

        for i, onset_frame in enumerate(onsets):
            fret_val = int(fret_indices[onset_frame, string_idx])
            
            # オンセット時にFRET_SILENCEが出ている場合は、前後1〜2フレームのフレットを参照して補完
            if fret_val == FRET_SILENCE_CLASS or fret_val < 0 or fret_val > MAX_FRETS:
                for offset in [-1, 1, -2, 2]:
                    f = onset_frame + offset
                    if 0 <= f < num_frames:
                        val = int(fret_indices[f, string_idx])
                        if val != FRET_SILENCE_CLASS and 0 <= val <= MAX_FRETS:
                            fret_val = val
                            break
            
            if fret_val != FRET_SILENCE_CLASS and 0 <= fret_val <= MAX_FRETS:
                next_onset = onsets[i+1] if i + 1 < len(onsets) else num_frames
                # 弦が鳴り続ける最大フレーム数 (約1.5秒)
                max_duration_frames = int(1.5 * SAMPLE_RATE / HOP_LENGTH)
                end_frame = min(onset_frame + max_duration_frames, next_onset)
                
                # 意図しない短すぎるノイズを除去（他モデルでいう duration 制約に相当）
                # しかしオンセットモデルなので判定幅は緩める
                pitch = tuning_pitches[string_idx] + fret_val
                onset_prob = float(onset_binary[onset_frame, string_idx])
                
                notes.append({
                    "start": round(onset_frame * time_per_frame, 4),
                    "end": round(end_frame * time_per_frame, 4),
                    "pitch": int(pitch),
                    "string": 6 - string_idx,  # MusicXML: 1=高E(1弦), 6=低E(6弦)
                    "fret": fret_val,
                    "velocity": min(0.5 + onset_prob * 0.5, 1.0),
                })

    # 時間順にソート
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def transcribe_guitar(wav_path: str, *, tuning_pitches: dict = None,
                      onset_threshold: float = 0.5,
                      chunk_duration: float = 30.0) -> dict:
    """
    WAVファイルからギターTAB情報を直接推定する。

    Parameters
    ----------
    wav_path : str
        入力WAVファイルのパス
    tuning_pitches : dict
        弦番号→MIDI番号マッピング
    onset_threshold : float
        オンセット検出の閾値 (0-1)
    chunk_duration : float
        処理チャンクの長さ（秒）。長い曲のメモリ管理用。

    Returns
    -------
    dict with keys:
        notes : list[dict] — 弦/フレット情報付きノートリスト
        total_notes : int
        duration : float — 曲の長さ（秒）
    """
    model, device = _load_model()

    print(f"[guitar_transcriber] Processing: {wav_path}")

    # CQTスペクトログラムを抽出
    cqt_features = _extract_cqt(wav_path)  # [3, N_BINS, T]
    channels, n_bins, total_frames = cqt_features.shape
    duration = total_frames * HOP_LENGTH / SAMPLE_RATE

    print(f"[guitar_transcriber] Duration: {duration:.1f}s, Frames: {total_frames}, Channels: {channels}")

    # チャンク単位で推論（メモリ管理）
    chunk_frames = int(chunk_duration * SAMPLE_RATE / HOP_LENGTH)
    all_onset_probs = []
    all_fret_preds = []

    with torch.no_grad():
        for start in range(0, total_frames, chunk_frames):
            end = min(start + chunk_frames, total_frames)
            chunk = cqt_features[:, :, start:end]  # [3, N_BINS, chunk_T]

            # バッチ次元を追加: [1, 3, N_BINS, chunk_T]
            x = chunk.unsqueeze(0).to(device)

            # モデル推論
            onset_logits, fret_logits = model(x)
            # onset_logits: [1, reduced_T, 6]
            # fret_logits: [1, reduced_T, 6, 22]

            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_preds = torch.argmax(fret_logits[0], dim=-1).cpu().numpy()

            all_onset_probs.append(onset_probs)
            all_fret_preds.append(fret_preds)

    # チャンクを結合
    onset_probs_full = np.concatenate(all_onset_probs, axis=0)  # [T', 6]
    fret_preds_full = np.concatenate(all_fret_preds, axis=0)    # [T', 6]

    print(f"[guitar_transcriber] Model output frames: {onset_probs_full.shape[0]}")

    # フレーム→ノートリストに変換
    notes = _frames_to_notes(
        onset_probs_full, fret_preds_full,
        tuning_pitches=tuning_pitches,
        onset_threshold=onset_threshold,
    )

    print(f"[guitar_transcriber] Detected {len(notes)} notes")

    return {
        "notes": notes,
        "total_notes": len(notes),
        "duration": duration,
    }


# --- フォールバック: Basic Pitch を使う旧方式 ---
def is_model_available() -> bool:
    """学習済みCRNNモデルが利用可能か確認する。"""
    model_path, config_path = _get_model_paths()
    return model_path is not None


if __name__ == "__main__":
    """テスト実行"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python guitar_transcriber.py <wav_path>")
        sys.exit(1)

    wav = sys.argv[1]
    result = transcribe_guitar(wav)
    print(f"\n=== Results ===")
    print(f"Notes: {result['total_notes']}")
    print(f"Duration: {result['duration']:.1f}s")
    print(f"Notes/sec: {result['total_notes']/result['duration']:.1f}")

    if result['notes']:
        # 弦・フレット分布
        from collections import Counter
        strings = Counter(n['string'] for n in result['notes'])
        frets = [n['fret'] for n in result['notes']]
        print(f"\nString distribution: {dict(sorted(strings.items()))}")
        print(f"Fret range: {min(frets)}-{max(frets)}")
        print(f"\nFirst 10 notes:")
        for n in result['notes'][:10]:
            print(f"  t={n['start']:.2f} s{n['string']} f{n['fret']} (MIDI {n['pitch']})")
