"""
synthtab_transcriber.py — SynthTab Model による TAB 転写
=========================================================
SynthTab (ICASSP 2024) の学習済みTabCNNモデルを使用して
音声ファイルからギターTAB (弦+フレット) を直接推定する。
6,700時間のデータで事前学習されたモデルを使用。
"""

import sys
import os
import torch
import numpy as np
import librosa
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# SynthTab model path — fine-tuned models are preferred over pretrained
SYNTHTAB_DIR = Path(__file__).parent.parent / "music-transcription" / "SynthTab" / "demo_embedding"

def _find_synthtab_model() -> Path:
    """SynthTabモデルを検索（fine-tuned優先）"""
    # 1. Fine-tuned models (generated/synthtab_finetune/models_*/model-*.pt)
    ft_dir = Path(__file__).parent.parent / "generated" / "synthtab_finetune"
    if ft_dir.exists():
        model_dirs = sorted(ft_dir.glob("models_*"), reverse=True)
        for md in model_dirs:
            models = sorted(md.glob("model-*.pt"), 
                          key=lambda p: int(p.stem.split('-')[1]))
            if models:
                return models[-1]  # highest iteration
    
    # 2. Pretrained finetuned on GuitarSet (original)
    orig = SYNTHTAB_DIR / "pretrained_models" / "finetuned" / "GuitarSet.pt"
    if orig.exists():
        return orig
    
    # 3. SynthTab pretrained (base)
    base = SYNTHTAB_DIR / "pretrained_models" / "SynthTab-Pretrained.pt"
    if base.exists():
        return base
    
    # 4. datasets dir fallback
    alt = Path(__file__).parent.parent / "datasets" / "SynthTab" / "demo_embedding" / "pretrained_models" / "SynthTab-Pretrained.pt"
    if alt.exists():
        return alt
    
    return orig  # fallback path (may not exist)

SYNTHTAB_MODEL_PATH = _find_synthtab_model()

# Audio parameters (must match SynthTab training)
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS_CQT = 192
BINS_PER_OCTAVE = 24

# Standard tuning MIDI notes (6th→1st string: E2→E4)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

# Cached model
_model = None
_device = None


def is_model_available() -> bool:
    """SynthTabモデルが利用可能かチェック"""
    return SYNTHTAB_MODEL_PATH.exists()


def _load_model():
    """SynthTabモデルをロード (キャッシュ)"""
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add SynthTab to path for tabcnn imports
    synth_path = str(SYNTHTAB_DIR)
    if synth_path not in sys.path:
        sys.path.insert(0, synth_path)

    _model = torch.load(str(SYNTHTAB_MODEL_PATH), map_location=_device, weights_only=False)
    if hasattr(_model, 'change_device'):
        _model.change_device(_device)
    _model.eval()

    # Online mode for inference
    if hasattr(_model, 'toggle_online') and not getattr(_model, 'online', False):
        _model.toggle_online()

    print(f"[synthtab] Model loaded on {_device}")
    return _model, _device


def _extract_cqt(audio: np.ndarray) -> np.ndarray:
    """CQT特徴量を抽出 (SynthTab仕様: 192 bins, 24 bpo)"""
    from amt_tools.features import CQT
    cqt_proc = CQT(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_bins=N_BINS_CQT,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    return cqt_proc.process_audio(audio)


def _frames_to_notes(
    tab_frames: np.ndarray,
    tuning: List[int] = None,
    min_note_frames: int = 2,
    merge_gap_frames: int = 2,
) -> List[Dict]:
    """
    フレームレベルのTAB出力をノートリストに変換する（改善版）。

    Parameters
    ----------
    tab_frames : np.ndarray, shape [T, 6]
        各フレームの各弦のフレット番号。-1 = 無音。
    tuning : list[int]
        開放弦のMIDIノート番号 [6弦, 5弦, ..., 1弦]
    min_note_frames : int
        ノートとして認識する最小フレーム数
    merge_gap_frames : int
        同弦・同フレットのノート間ギャップがこのフレーム数以下なら統合

    Returns
    -------
    list[dict]
        各ノートの {start, end, pitch, string, fret, velocity}
    """
    if tuning is None:
        tuning = STANDARD_TUNING

    time_per_frame = HOP_LENGTH / SAMPLE_RATE
    min_note_ms = 40  # 最小ノート長 (ms)
    min_note_len = max(min_note_frames, int(min_note_ms / 1000 / time_per_frame))

    notes = []
    num_strings = tab_frames.shape[1]

    for string_idx in range(num_strings):
        frets = tab_frames[:, string_idx].copy()
        open_pitch = tuning[string_idx]

        # Step 1: メディアンフィルタで1-2フレームの揺れを除去
        if len(frets) >= 3:
            smoothed = frets.copy()
            for t in range(1, len(frets) - 1):
                if frets[t] >= 0 and frets[t-1] == frets[t+1] and frets[t] != frets[t-1]:
                    smoothed[t] = frets[t-1]  # 前後と同じに修正
            frets = smoothed

        # Step 2: Run-length encoding でノート区間を抽出
        raw_notes = []
        in_note = False
        note_start = 0
        note_fret = -1

        for t in range(len(frets)):
            current_fret = int(frets[t])

            if current_fret >= 0:
                if not in_note or current_fret != note_fret:
                    if in_note:
                        raw_notes.append((note_start, t, note_fret))
                    in_note = True
                    note_start = t
                    note_fret = current_fret
            else:
                if in_note:
                    raw_notes.append((note_start, t, note_fret))
                in_note = False

        if in_note:
            raw_notes.append((note_start, len(frets), note_fret))

        # Step 3: 同弦・同フレットの近接ノートを統合
        merged = []
        for start, end, fret in raw_notes:
            if merged and merged[-1][2] == fret and (start - merged[-1][1]) <= merge_gap_frames:
                merged[-1] = (merged[-1][0], end, fret)
            else:
                merged.append((start, end, fret))

        # Step 4: 最小長フィルタ適用 & ノート生成
        for start, end, fret in merged:
            duration_frames = end - start
            if duration_frames < min_note_len:
                continue

            midi_pitch = open_pitch + fret
            # ギター音域外を除外 (MIDI 40-96)
            if midi_pitch < 38 or midi_pitch > 96:
                continue

            notes.append({
                "start": round(start * time_per_frame, 4),
                "end": round(end * time_per_frame, 4),
                "pitch": midi_pitch,
                "string": 6 - string_idx,  # MusicXML: 1=高E, 6=低E
                "fret": fret,
                "velocity": 0.8,  # SynthTabは velocity を出力しないのでデフォルト
            })

    # Sort by start time
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def transcribe_guitar(
    wav_path: str,
    tuning_pitches: Optional[Dict[int, int]] = None,
    chunk_seconds: float = 30.0,
) -> Dict:
    """
    SynthTabモデルを使ってギターTABを生成する。

    Parameters
    ----------
    wav_path : str
        WAVファイルのパス
    tuning_pitches : dict, optional
        弦番号→MIDI {0: 40, 1: 45, ...}
    chunk_seconds : float
        チャンク処理の秒数

    Returns
    -------
    dict
        {"notes": [...], "method": "synthtab"}
    """
    import amt_tools.tools as tools

    model, device = _load_model()

    # Tuning
    if tuning_pitches:
        tuning = [tuning_pitches[i] for i in range(6)]
    else:
        tuning = STANDARD_TUNING

    # Load audio
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    total_duration = len(audio) / SAMPLE_RATE
    print(f"[synthtab] Audio: {total_duration:.1f}s")

    # Process in chunks to manage memory
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    # Add overlap for continuity
    overlap_samples = int(2.0 * SAMPLE_RATE)
    all_tab_frames = []

    pos = 0
    chunk_idx = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]

        print(f"[synthtab] Processing chunk {chunk_idx} ({pos/SAMPLE_RATE:.0f}-{end/SAMPLE_RATE:.0f}s)")

        # Extract CQT
        features = _extract_cqt(chunk)

        # Run model
        batch = {tools.KEY_FEATS: tools.array_to_tensor(features, device)}
        batch = model.pre_proc(batch)

        with torch.no_grad():
            output = model(batch[tools.KEY_FEATS])

        batch[tools.KEY_OUTPUT] = output
        output = model.post_proc(batch)

        tab = output[tools.KEY_TABLATURE].cpu().numpy()  # [T, 6, 1]
        tab = tab.squeeze(-1)  # [T, 6]

        # Handle overlap: skip first overlap frames for non-first chunks
        if pos > 0 and overlap_samples > 0:
            overlap_frames = int(overlap_samples / SAMPLE_RATE * SAMPLE_RATE / HOP_LENGTH)
            tab = tab[overlap_frames:]

        all_tab_frames.append(tab)

        pos = end - (overlap_samples if end < len(audio) else 0)
        chunk_idx += 1

    # Concatenate all chunks
    tab_frames = np.concatenate(all_tab_frames, axis=0)
    print(f"[synthtab] Total frames: {tab_frames.shape[0]}")

    # Convert frames to notes
    notes = _frames_to_notes(tab_frames, tuning=tuning, min_note_frames=3)
    print(f"[synthtab] Detected {len(notes)} notes")

    return {
        "notes": notes,
        "total_count": len(notes),
        "method": "synthtab",
    }


# Quick test
if __name__ == "__main__":
    if not is_model_available():
        print("SynthTab model not found!")
        sys.exit(1)

    test_wav = r'D:\Music\nextchord-solotab\uploads\20260223-213250-yt-f30188\converted.wav'
    result = transcribe_guitar(test_wav)
    print(f"\nTotal notes: {result['total_count']}")
    print("First 30 notes:")
    for n in result['notes'][:30]:
        print(f"  t={n['start']:.2f}s s{n['string']} f{n['fret']} (MIDI {n['pitch']})")
