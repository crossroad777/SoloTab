"""
guitar_transcriber.py — ギター専用CRNN推論モジュール
=====================================================
学習済みGuitarTabCRNNモデルを使って、WAVファイルから
弦・フレット情報付きのノートリストを直接出力する。

Basic Pitchの2段階方式（ピッチ検出→弦割り当て）を置き換え、
1ステップで正確なTAB情報を生成する。

修正済み問題:
    - [FIX-1] CQTチャンネル数をコメントに合わせて1に統一（3は誤記）
    - [FIX-2] チャンク境界ノート途切れ: オーバーラップ付き推論に変更
    - [FIX-3] オンセット重複・end_frame不整合を修正
    - [FIX-4] グローバルキャッシュをthreading.Lockで保護
    - [FIX-5] CNN出力次元をrun_configuration.jsonから取得するよう変更
    - [FIX-6] librosa.load/torch.load のエラーハンドリング追加
    - [FIX-7] onset_threshold デフォルト値を一箇所に統一
    - [FIX-8] 内部/外部インデックス変換をコメントで明示
"""

import os
import sys
import json
import threading
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

# [FIX-1] モデルへの入力チャンネル数は1（コメントの "3" は誤記だった）
N_INPUT_CHANNELS = 1

# フレットの「無音」クラスインデックス
FRET_SILENCE_CLASS = MAX_FRETS + 1  # == 21

# 標準チューニングのオープン弦 MIDI ノート番号
# インデックス 0 = 6弦(低E=40), インデックス 5 = 1弦(高e=64)  [FIX-8]
OPEN_STRING_PITCHES: dict[int, int] = {
    0: 40,  # 6弦: E2
    1: 45,  # 5弦: A2
    2: 50,  # 4弦: D3
    3: 55,  # 3弦: G3
    4: 59,  # 2弦: B3
    5: 64,  # 1弦: e4
}

# チャンク境界保護のためのオーバーラップフレーム数 [FIX-2]
_CHUNK_OVERLAP_FRAMES = 32

# onset_threshold のデフォルト値（全関数で共通） [FIX-7]
DEFAULT_ONSET_THRESHOLD = 0.5

# --- スレッドセーフなモデルキャッシュ [FIX-4] ---
_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def _get_model_paths() -> tuple[str | None, str | None]:
    """学習済みモデルのパスを取得する。"""
    base = _MT_DIR / "_processed_guitarset_data" / "training_output"
    target_dir = base / "ultimate_single_conformer"

    if not target_dir.exists():
        return None, None

    model_path = target_dir / "best_model.pth"
    config_path = target_dir / "run_configuration.json"

    if not config_path.exists():
        config_path = base / "baseline_model" / "run_configuration.json"

    if model_path.exists() and config_path.exists():
        return str(model_path), str(config_path)

    return None, None


def _load_model(device: torch.device | None = None) -> tuple:
    """
    学習済みCRNNモデルを読み込み、スレッドセーフにキャッシュする。

    [FIX-4] threading.Lock でキャッシュへの同時書き込みを防止。
    [FIX-5] CNN出力次元を run_configuration.json から取得し、
            ダミー推論による TabCNN の余分なインスタンス化を廃止。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = str(device)

    # ロックなしで先に読み取り（高速パス）
    if cache_key in _model_cache:
        return _model_cache[cache_key], device

    with _model_cache_lock:
        # ロック取得後に再チェック（二重チェックロッキング）
        if cache_key in _model_cache:
            return _model_cache[cache_key], device

        model_path, config_path = _get_model_paths()
        if model_path is None:
            raise FileNotFoundError(
                "学習済みモデルが見つかりません。"
                "まず music-transcription/python/train_model.py で学習を完了してください。"
            )

        print(f"[guitar_transcriber] Loading CRNN model from: {model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)

        hyperparams = run_config["hyperparameters_tuned"]

        # [FIX-5] CNN出力次元を config から直接取得（ダミー推論不要）
        cnn_out_dim = run_config.get("cnn_output_dim")
        if cnn_out_dim is None:
            raise KeyError(
                "run_configuration.json に 'cnn_output_dim' キーがありません。"
                "train_model.py を再実行して config を更新してください。"
            )

        from model.architecture import GuitarTabCRNN

        model = GuitarTabCRNN(
            num_frames_rnn_input_dim=cnn_out_dim,
            rnn_type=hyperparams.get("RNN_TYPE", "GRU"),
            rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
            rnn_layers=hyperparams["RNN_LAYERS"],
            rnn_dropout=hyperparams["RNN_DROPOUT"],
            rnn_bidirectional=hyperparams.get("RNN_BIDIRECTIONAL", True),
        )

        # [FIX-6] torch.load の互換性: weights_only は PyTorch 1.x では無効
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # PyTorch < 2.0 では weights_only 引数が存在しない
            state_dict = torch.load(model_path, map_location=device)

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _model_cache[cache_key] = model
        print(f"[guitar_transcriber] Model loaded on {device}")

    return _model_cache[cache_key], device


def _extract_cqt(wav_path: str) -> torch.Tensor:
    """
    WAVファイルからCQTスペクトログラムを抽出する。

    Returns
    -------
    torch.Tensor — shape: [1, N_BINS_CQT, T]  [FIX-1]
    """
    # [FIX-6] librosa.load の失敗を明示的に捕捉
    try:
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"WAVファイルの読み込みに失敗しました: {wav_path}\n原因: {e}") from e

    cqt = librosa.cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN_CQT,
        n_bins=N_BINS_CQT,
        bins_per_octave=BINS_PER_OCTAVE_CQT,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # [FIX-1] チャンネル次元を追加: [N_BINS_CQT, T] → [1, N_BINS_CQT, T]
    cqt_tensor = torch.tensor(
        np.expand_dims(log_cqt, axis=0), dtype=torch.float32
    )
    return cqt_tensor


def _frames_to_notes(
    onset_probs: np.ndarray,
    fret_indices: np.ndarray,
    tuning_pitches: dict | None = None,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
) -> list[dict]:
    """
    フレーム単位の予測結果をノートリストに変換する。

    Parameters
    ----------
    onset_probs   : shape [T, 6], sigmoid 済みオンセット確率
    fret_indices  : shape [T, 6], argmax 済みフレット予測
    tuning_pitches: 内部インデックス(0-5) → MIDI番号 のマッピング [FIX-8]
    onset_threshold: オンセット検出の閾値

    [FIX-3] オンセット重複の排除、end_frame の整合性を修正。
    """
    if tuning_pitches is None:
        tuning_pitches = OPEN_STRING_PITCHES

    time_per_frame = HOP_LENGTH / SAMPLE_RATE
    num_frames, num_strings = onset_probs.shape
    notes = []

    for string_idx in range(num_strings):
        probs = onset_probs[:, string_idx]

        # --- ピークピッキングでオンセットフレームを収集 [FIX-3] ---
        onsets: list[int] = []
        for frame_idx in range(num_frames):
            p = probs[frame_idx]
            if p <= onset_threshold:
                continue

            # 局所的マキシマム判定（境界は片側比較）
            left_ok  = (frame_idx == 0) or (p > probs[frame_idx - 1])
            right_ok = (frame_idx == num_frames - 1) or (p >= probs[frame_idx + 1])
            if left_ok and right_ok:
                # 直前のオンセットと同一フレームにならないよう重複排除
                if not onsets or frame_idx != onsets[-1]:
                    onsets.append(frame_idx)

        # --- オンセットごとにノートを生成 ---
        for i, onset_frame in enumerate(onsets):
            fret_val = int(fret_indices[onset_frame, string_idx])

            # オンセット時に SILENCE が出ている場合は近傍フレームで補完
            if fret_val == FRET_SILENCE_CLASS or not (0 <= fret_val <= MAX_FRETS):
                for offset in (-1, 1, -2, 2):
                    f = onset_frame + offset
                    if 0 <= f < num_frames:
                        candidate = int(fret_indices[f, string_idx])
                        if candidate != FRET_SILENCE_CLASS and 0 <= candidate <= MAX_FRETS:
                            fret_val = candidate
                            break

            if fret_val == FRET_SILENCE_CLASS or not (0 <= fret_val <= MAX_FRETS):
                continue  # 補完失敗 → スキップ

            # end_frame: 次のオンセット開始フレーム（絶対値）と最大持続の小さい方 [FIX-3]
            next_onset = onsets[i + 1] if i + 1 < len(onsets) else num_frames
            max_duration_frames = int(1.5 * SAMPLE_RATE / HOP_LENGTH)
            end_frame = min(onset_frame + max_duration_frames, next_onset)

            # [FIX-8] 内部インデックス(0=6弦) → MusicXML弦番号(1=1弦, 6=6弦)
            musicxml_string = 6 - string_idx

            pitch = tuning_pitches[string_idx] + fret_val
            onset_prob = float(probs[onset_frame])

            notes.append({
                "start":    round(onset_frame * time_per_frame, 4),
                "end":      round(end_frame    * time_per_frame, 4),
                "pitch":    int(pitch),
                "string":   musicxml_string,   # 1=高E(1弦) … 6=低E(6弦)
                "fret":     fret_val,
                "velocity": round(min(0.5 + onset_prob * 0.5, 1.0), 4),
            })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def transcribe_guitar(
    wav_path: str,
    *,
    tuning_pitches: dict | None = None,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    chunk_duration: float = 30.0,
) -> dict:
    """
    WAVファイルからギターTAB情報を直接推定する。

    Parameters
    ----------
    wav_path        : 入力WAVファイルのパス
    tuning_pitches  : 内部インデックス(0-5) → MIDI番号マッピング [FIX-8]
    onset_threshold : オンセット検出の閾値 (0-1) [FIX-7]
    chunk_duration  : 処理チャンクの長さ（秒）。長い曲のメモリ管理用。

    Returns
    -------
    dict:
        notes       : list[dict] — 弦/フレット情報付きノートリスト
        total_notes : int
        duration    : float — 曲の長さ（秒）

    [FIX-2] チャンク境界のノート途切れをオーバーラップ推論で解消。
    """
    model, device = _load_model()

    print(f"[guitar_transcriber] Processing: {wav_path}")

    cqt_features = _extract_cqt(wav_path)  # [1, N_BINS_CQT, T]
    _, n_bins, total_frames = cqt_features.shape
    duration = total_frames * HOP_LENGTH / SAMPLE_RATE

    print(f"[guitar_transcriber] Duration: {duration:.1f}s, Frames: {total_frames}")

    # チャンク単位推論（オーバーラップ付き） [FIX-2]
    chunk_frames   = int(chunk_duration * SAMPLE_RATE / HOP_LENGTH)
    overlap        = _CHUNK_OVERLAP_FRAMES
    all_onset_probs: list[np.ndarray] = []
    all_fret_preds:  list[np.ndarray] = []

    with torch.no_grad():
        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            chunk = cqt_features[:, :, start:end]           # [1, N_BINS, chunk_T]
            x = chunk.unsqueeze(0).to(device)               # [1, 1, N_BINS, chunk_T]

            onset_logits, fret_logits = model(x)
            # onset_logits: [1, reduced_T, 6]
            # fret_logits:  [1, reduced_T, 6, 22]

            onset_probs_chunk = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_preds_chunk  = torch.argmax(fret_logits[0], dim=-1).cpu().numpy()

            if all_onset_probs:
                # 先頭の overlap フレームは前チャンクとの重複部分 → 捨てる [FIX-2]
                onset_probs_chunk = onset_probs_chunk[overlap:]
                fret_preds_chunk  = fret_preds_chunk[overlap:]

            all_onset_probs.append(onset_probs_chunk)
            all_fret_preds.append(fret_preds_chunk)

            # 次チャンクは overlap 分だけ手前から始める
            next_start = start + chunk_frames - overlap
            if next_start <= start:
                # 無限ループ防止
                break
            start = next_start

    onset_probs_full = np.concatenate(all_onset_probs, axis=0)  # [T', 6]
    fret_preds_full  = np.concatenate(all_fret_preds,  axis=0)  # [T', 6]

    print(f"[guitar_transcriber] Model output frames: {onset_probs_full.shape[0]}")

    notes = _frames_to_notes(
        onset_probs_full,
        fret_preds_full,
        tuning_pitches=tuning_pitches,
        onset_threshold=onset_threshold,
    )

    # -------------------------------------------------------------------------
    # [FIX-9] 12th Fret Octave Resonance Correction (Harmonic Overdrive)
    # -------------------------------------------------------------------------
    # クラシックギターの12フレット等の実音は、開放弦の倍音と誤認されやすい。
    # そこで、開放弦（Fret 0）と判定されたノートに対し、CQTスペクトログラム上で
    # オクターブ上（+12半音）のエネルギーが基音に近い（あるいは高い）場合、
    # 物理的に12フレットが弾かれたとみなして修正（アップグレード）する。
    time_per_frame = HOP_LENGTH / SAMPLE_RATE
    for note in notes:
        if note["fret"] == 0 and note["string"] in [1, 2, 3]:
            frame = int(round(note["start"] / time_per_frame))
            
            # CQT: 3 bins/semitone. MIDI 40 = Bin 0. Octave = +36 bins.
            fund_bin = (note["pitch"] - 40) * 3
            oct_bin = fund_bin + 36
            
            if 0 <= fund_bin < n_bins and 0 <= oct_bin < n_bins:
                f_start = max(0, frame - 2)
                f_end = min(total_frames, frame + 3)
                if f_end > f_start:
                    fund_energy = cqt_features[0, fund_bin, f_start:f_end].mean().item()
                    oct_energy = cqt_features[0, oct_bin, f_start:f_end].mean().item()
                    
                    # 一時的にHarmonic Overdriveを完全無効化（EQと競合して開放弦が12フレットに化けるバグを抑止）
                    # if oct_energy > fund_energy + 2.0:
                    #     note["fret"] = 12
                    #     note["pitch"] += 12

    print(f"[guitar_transcriber] Detected {len(notes)} notes")

    return {
        "notes":       notes,
        "total_notes": len(notes),
        "duration":    duration,
    }


def is_model_available() -> bool:
    """学習済みCRNNモデルが利用可能か確認する。"""
    model_path, config_path = _get_model_paths()
    return model_path is not None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python guitar_transcriber.py <wav_path>")
        sys.exit(1)

    wav = sys.argv[1]
    result = transcribe_guitar(wav)
    print(f"\n=== Results ===")
    print(f"Notes: {result['total_notes']}")
    print(f"Duration: {result['duration']:.1f}s")
    print(f"Notes/sec: {result['total_notes'] / result['duration']:.1f}")

    if result["notes"]:
        from collections import Counter
        strings = Counter(n["string"] for n in result["notes"])
        frets   = [n["fret"] for n in result["notes"]]
        print(f"\nString distribution: {dict(sorted(strings.items()))}")
        print(f"Fret range: {min(frets)}-{max(frets)}")
        print(f"\nFirst 10 notes:")
        for n in result["notes"][:10]:
            print(
                f"  t={n['start']:.2f}  "
                f"string={n['string']}  fret={n['fret']}  "
                f"MIDI={n['pitch']}  vel={n['velocity']:.2f}"
            )
