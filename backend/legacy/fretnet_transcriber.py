"""
fretnet_transcriber.py — FretNet モデルによるギターTAB転写
==========================================================
FretNet (ISMIR 2022) の学習済みモデルを使用して
音声ファイルからギターTAB (弦+フレット) を推定する。
連続値ピッチ推定により、ビブラートやスライド等の表現も捕捉可能。
"""

# Windows cp932 encoding fix — amt_tools内のUnicode文字対策
import io, sys
if sys.stdout and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower().replace('-','') != 'utf8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import os
import torch
import numpy as np
import librosa
from typing import List, Dict, Optional
from pathlib import Path

# Model search paths
FRETNET_MODEL_DIRS = [
    Path(__file__).parent.parent / "generated" / "fretnet_models" / "models",
    Path(__file__).parent / "fretnet_models" / "models",
    Path(os.path.expanduser("~")) / "Downloads" / "FretNet" / "models",
]

# IDMT fine-tuned model (single model, used as highest priority)
IDMT_FINETUNE_DIR = Path(__file__).parent.parent / "generated" / "fretnet_idmt_finetune" / "models" / "idmt-finetune"

# SCORE-SET fine-tuned model (second priority)
SCORESET_FINETUNE_DIR = Path(__file__).parent.parent / "generated" / "fretnet_scoreset_finetune" / "models" / "scoreset-finetune"

# SoloTab Baseline Model (Acoustic Mixed Fine-tuned: GuitarSet + GAPS)
BASELINE_MODEL_DIR = Path(__file__).parent.parent / "generated" / "baseline_model_finetune"

# 3-stage trained model (SynthTab pretrain + GuitarSet/GAPS finetune + technique FT)
THREESTAGE_S3_DIR = Path(__file__).parent.parent / "generated" / "fretnet_3stage" / "stage3_technique" / "fold-0"
THREESTAGE_S2_DIR = Path(__file__).parent.parent / "generated" / "fretnet_3stage" / "stage2_finetuned" / "fold-0"

# GAPS fine-tuned model (standalone Stage 2)
GAPS_FINETUNE_DIR = Path(__file__).parent.parent / "generated" / "fretnet_gaps_finetune" / "models"

# Two-stage model (synth pretrain + GuitarSet finetune)
TWOSTAGE_DIR = Path(__file__).parent.parent / "generated" / "fretnet_twostage" / "models"

# IDMT fine-tuned model (Legacy)
IDMT_DIR = Path(__file__).parent.parent / "generated" / "fretnet_idmt" / "models"
SAMPLE_RATE = 22050
HOP_LENGTH = 512

# Standard tuning MIDI notes
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

_models = {}  # fold_id -> model
_device = None


def find_all_models() -> Dict[str, Path]:
    """
    全foldの学習済みFretNetモデルを検索。
    優先順: Baseline Model > 3-stage S3 > 3-stage S2 > Two-stage > IDMT fine-tuned > SCORE-SET fine-tuned > GAPS finetuned > 通常モデル
    Returns: {"fold-0": Path, "fold-1": Path, ...}
    """
    fold_models = {}
    
    # SoloTab Baseline Model — 最優先
    if BASELINE_MODEL_DIR.exists():
        best_path = None
        for run_dir in BASELINE_MODEL_DIR.glob("run_*"):
            if (run_dir / "best_model.pth").exists():
                best_path = run_dir / "best_model.pth"
                break
        if best_path:
            for i in range(6):
                fold_models[f"fold-{i}"] = best_path
            return fold_models
    
    # 3-stage model: Stage 3 (technique FT)
    for stage_dir in [THREESTAGE_S3_DIR, THREESTAGE_S2_DIR]:
        if stage_dir.exists():
            best_iter = -1
            best_path = None
            for model_file in stage_dir.glob("model-*.pt"):
                try:
                    iteration = int(model_file.stem.split('-')[1])
                    if iteration > best_iter:
                        best_iter = iteration
                        best_path = model_file
                except (IndexError, ValueError):
                    continue
            if best_path:
                for i in range(6):
                    fold_models[f"fold-{i}"] = best_path
                return fold_models
    
    # GAPS fine-tuned model (standalone Stage 2)
    if GAPS_FINETUNE_DIR.exists():
        best_iter = -1
        best_path = None
        for model_file in GAPS_FINETUNE_DIR.glob("model-*.pt"):
            try:
                iteration = int(model_file.stem.split('-')[1])
                if iteration > best_iter:
                    best_iter = iteration
                    best_path = model_file
            except (IndexError, ValueError):
                continue
        if best_path:
            for i in range(6):
                fold_models[f"fold-{i}"] = best_path
            return fold_models
    
    # Two-stage model (synth pretrain + GuitarSet finetune)
    if TWOSTAGE_DIR.exists():
        best_iter = -1
        best_path = None
        for model_file in TWOSTAGE_DIR.glob("model-*.pt"):
            try:
                iteration = int(model_file.stem.split('-')[1])
                if iteration > best_iter:
                    best_iter = iteration
                    best_path = model_file
            except (IndexError, ValueError):
                continue
        if best_path:
            for i in range(6):
                fold_models[f"fold-{i}"] = best_path
            return fold_models
    
    # IDMT fine-tuned モデルを検索
    if IDMT_FINETUNE_DIR.exists():
        best_iter = -1
        best_path = None
        for model_file in IDMT_FINETUNE_DIR.glob("model-*.pt"):
            try:
                iteration = int(model_file.stem.split('-')[1])
                if iteration > best_iter:
                    best_iter = iteration
                    best_path = model_file
            except (IndexError, ValueError):
                continue
        if best_path:
            # IDMT fine-tunedは単一モデル → 全foldで共有
            for i in range(6):
                fold_models[f"fold-{i}"] = best_path
            return fold_models
    
    # SCORE-SET fine-tuned モデルを2番目に検索
    if SCORESET_FINETUNE_DIR.exists():
        best_iter = -1
        best_path = None
        for model_file in SCORESET_FINETUNE_DIR.glob("model-*.pt"):
            try:
                iteration = int(model_file.stem.split('-')[1])
                if iteration > best_iter:
                    best_iter = iteration
                    best_path = model_file
            except (IndexError, ValueError):
                continue
        if best_path:
            # SCORE-SET fine-tunedは単一モデル → 全foldで共有
            for i in range(6):
                fold_models[f"fold-{i}"] = best_path
            return fold_models
    
    for model_dir in FRETNET_MODEL_DIRS:
        if not model_dir.exists():
            continue
        
        # まず各foldの gaps_finetuned サブフォルダを確認
        for fold_dir in sorted(model_dir.glob("fold-*")):
            fold_name = fold_dir.name
            if fold_name in fold_models:
                continue
            
            # GAPS finetunedモデルを優先
            gaps_dir = fold_dir / "gaps_finetuned"
            best_iter = -1
            best_path = None
            
            if gaps_dir.exists():
                for model_file in gaps_dir.glob("model-gaps-*.pt"):
                    try:
                        iteration = int(model_file.stem.split('-')[2])
                        if iteration > best_iter:
                            best_iter = iteration
                            best_path = model_file
                    except (IndexError, ValueError):
                        continue
            
            # GAPSモデルが見つからなければ通常モデル
            if not best_path:
                for model_file in fold_dir.glob("model-*.pt"):
                    try:
                        iteration = int(model_file.stem.split('-')[1])
                        if iteration > best_iter:
                            best_iter = iteration
                            best_path = model_file
                    except (IndexError, ValueError):
                        continue
            
            if best_path:
                fold_models[fold_name] = best_path
        
        # グローバルな gaps_finetuned フォルダも確認 (fold-0用、上書き優先)
        global_gaps = model_dir / "gaps_finetuned"
        if global_gaps.exists():
            best_iter = -1
            best_path = None
            for model_file in global_gaps.glob("model-gaps-*.pt"):
                try:
                    iteration = int(model_file.stem.split('-')[2])
                    if iteration > best_iter:
                        best_iter = iteration
                        best_path = model_file
                except (IndexError, ValueError):
                    continue
            if best_path:
                fold_models["fold-0"] = best_path  # GAPS ftを優先上書き
    
    return fold_models


def find_model() -> Optional[Path]:
    """後方互換: 最初のfoldモデルを返す"""
    models = find_all_models()
    return next(iter(models.values()), None) if models else None


def is_model_available() -> bool:
    """FretNetモデルが利用可能かチェック"""
    return len(find_all_models()) > 0


def _load_model(model_path: Path = None):
    """FretNetモデルをロード (キャッシュ)"""
    global _models, _device
    
    if model_path is None:
        model_path = find_model()
    if model_path is None:
        raise FileNotFoundError("FretNet model not found.")
    
    path_str = str(model_path)
    if path_str in _models:
        return _models[path_str], _device

    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(path_str, map_location=_device, weights_only=False)
    if hasattr(model, 'change_device'):
        gpu_id = 0 if torch.cuda.is_available() else -1
        model.change_device(gpu_id)
    model.eval()
    
    _models[path_str] = model
    print(f"[fretnet] Model loaded: {model_path.parent.name}/{model_path.name} on {_device}")
    return model, _device


def _run_single_fold(model, wav_path, tuning, tools_mod, HCQT_cls,
                     ComboEstimator_cls, TablatureWrapper_cls,
                     StackedOffsetsWrapper_cls, StackedNoteTranscriber_cls,
                     StackedPitchListTablatureWrapper_cls, utils_mod,
                     features):
    """1つのfoldモデルで推論を実行。"""
    from amt_tools.inference import run_offline
    import numpy as np
    profile = model.profile

    estimator = ComboEstimator_cls([
        TablatureWrapper_cls(profile=profile),
        StackedOffsetsWrapper_cls(profile=profile),
        StackedNoteTranscriber_cls(profile=profile),
        StackedPitchListTablatureWrapper_cls(
            profile=profile,
            multi_pitch_key=tools_mod.KEY_TABLATURE,
            multi_pitch_rel_key=utils_mod.KEY_TABLATURE_REL)
    ])

    predictions = run_offline(features, model, estimator)

    # onsetsが全ゼロの場合のフォールバック:
    # multi_pitchからonsetsを再計算してノート抽出をやり直す
    onsets = predictions.get(tools_mod.KEY_ONSETS)
    stacked_notes = predictions[tools_mod.KEY_NOTES]
    total_notes = sum(len(stacked_notes.get(s, ([], []))[0]) for s in range(6))

    if total_notes == 0 and onsets is not None and isinstance(onsets, np.ndarray) and onsets.max() == 0:
        print(f"[fretnet] Onsets all zero - using multi_pitch fallback")
        multi_pitch = predictions.get(tools_mod.KEY_MULTIPITCH)
        times = predictions.get(tools_mod.KEY_TIMES)
        if multi_pitch is not None and times is not None:
            stacked_notes = {}
            offsets_all = predictions.get(tools_mod.KEY_OFFSETS)
            for slc in range(multi_pitch.shape[0]):
                mp = multi_pitch[slc]
                # multi_pitchからonsetsを生成
                derived_onsets = tools_mod.multi_pitch_to_onsets(mp)
                off = offsets_all[slc] if offsets_all is not None else None
                pitches, intervals = tools_mod.multi_pitch_to_notes(mp, times, profile, derived_onsets, off)
                stacked_notes.update(tools_mod.notes_to_stacked_notes(pitches, intervals, slc))
            total_notes = sum(len(stacked_notes.get(s, ([], []))[0]) for s in range(6))
            print(f"[fretnet] Fallback recovered {total_notes} notes")

    notes = []
    for string_idx in range(6):
        if string_idx not in stacked_notes:
            continue
        open_pitch = tuning[string_idx]
        pitches, intervals = stacked_notes[string_idx]
        if len(pitches) == 0 or len(intervals) == 0:
            continue
        for i in range(len(pitches)):
            midi_pitch = int(pitches[i])
            if midi_pitch < 23:
                fret = midi_pitch
                midi_pitch = open_pitch + fret
            else:
                fret = midi_pitch - open_pitch
            if fret < 0 or fret > 22:
                continue
            onset = float(intervals[i][0])
            offset = float(intervals[i][1])
            notes.append({
                "start": round(onset, 4),
                "end": round(offset, 4),
                "pitch": midi_pitch,
                "string": 6 - string_idx,
                "fret": fret,
                "velocity": 0.8,
            })
    return notes


def transcribe_guitar(
    wav_path: str,
    tuning_pitches: Optional[Dict[int, int]] = None,
) -> Dict:
    """
    FretNet全foldアンサンブルでギターTABを生成する。
    各foldモデルの結果を統合し、2+foldが合意したノートを採用。
    弦/フレット情報はFretNetの出力をそのまま使用。
    """
    from amt_tools.features import HCQT
    from amt_tools.transcribe import (ComboEstimator, TablatureWrapper,
                                       StackedOffsetsWrapper, StackedNoteTranscriber)
    from guitar_transcription_continuous.estimators import StackedPitchListTablatureWrapper
    import guitar_transcription_continuous.utils as utils
    import amt_tools.tools as tools

    # Tuning
    if tuning_pitches:
        tuning = [tuning_pitches[i] for i in range(6)]
    else:
        tuning = STANDARD_TUNING

    # Load audio (1回だけ)
    audio, _ = tools.load_normalize_audio(wav_path, SAMPLE_RATE)
    total_duration = len(audio) / SAMPLE_RATE
    print(f"[fretnet] Audio: {total_duration:.1f}s")

    # HCQT features (1回だけ計算)
    data_proc = HCQT(sample_rate=SAMPLE_RATE,
                     hop_length=HOP_LENGTH,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)
    features = {
        tools.KEY_FEATS: data_proc.process_audio(audio),
        tools.KEY_TIMES: data_proc.get_times(audio)
    }

    # 全foldモデルで推論
    fold_models = find_all_models()
    if not fold_models:
        raise FileNotFoundError("FretNet model not found.")
    
    all_fold_notes = []  # [fold_notes_list, ...]
    for fold_name, model_path in sorted(fold_models.items()):
        print(f"[fretnet] Running {fold_name}...")
        model, device = _load_model(model_path)
        fold_notes = _run_single_fold(
            model, wav_path, tuning, tools, HCQT,
            ComboEstimator, TablatureWrapper,
            StackedOffsetsWrapper, StackedNoteTranscriber,
            StackedPitchListTablatureWrapper, utils,
            features
        )
        all_fold_notes.append(fold_notes)
        print(f"[fretnet]   {fold_name}: {len(fold_notes)} notes")

    # --- 全foldの結果をマージ ---
    # 同一ノート判定: time ±80ms, pitch ±1半音
    TIME_TOL = 0.08
    PITCH_TOL = 1
    
    # 全ノートをフラットに + fold情報付加
    all_notes = []
    for fold_idx, fold_notes in enumerate(all_fold_notes):
        for n in fold_notes:
            nc = dict(n)
            nc["_fold"] = fold_idx
            all_notes.append(nc)
    
    all_notes.sort(key=lambda n: (n["start"], n["pitch"]))
    
    # グリーディマッチング
    used = [False] * len(all_notes)
    groups = []
    for i in range(len(all_notes)):
        if used[i]:
            continue
        group = [all_notes[i]]
        used[i] = True
        folds_in_group = {all_notes[i]["_fold"]}
        for j in range(i + 1, len(all_notes)):
            if used[j]:
                continue
            if all_notes[j]["start"] - all_notes[i]["start"] > TIME_TOL * 2:
                break
            if all_notes[j]["_fold"] in folds_in_group:
                continue
            if (abs(all_notes[j]["start"] - all_notes[i]["start"]) <= TIME_TOL and
                abs(all_notes[j]["pitch"] - all_notes[i]["pitch"]) <= PITCH_TOL):
                group.append(all_notes[j])
                used[j] = True
                folds_in_group.add(all_notes[j]["_fold"])
        groups.append(group)
    
    # min_folds=1: 1fold以上が検出すれば採用 (FretNet内部アンサンブルなので寛容に)
    # ただし2+foldが合意したノートに高信頼度を付与
    min_folds = 1
    notes = []
    for group in groups:
        if len(group) >= min_folds:
            # 最も多くのfoldで合意した弦/フレットを採用
            avg_start = sum(n["start"] for n in group) / len(group)
            avg_end = sum(n["end"] for n in group) / len(group)
            # ピッチはモード(最頻値)を使用
            from collections import Counter
            pitch_counts = Counter(n["pitch"] for n in group)
            best_pitch = pitch_counts.most_common(1)[0][0]
            # 弦/フレットも最頻値
            sf_counts = Counter((n["string"], n["fret"]) for n in group)
            best_sf = sf_counts.most_common(1)[0][0]
            
            notes.append({
                "start": round(avg_start, 4),
                "end": round(avg_end, 4),
                "pitch": best_pitch,
                "string": best_sf[0],
                "fret": best_sf[1],
                "velocity": min(0.5 + len(group) * 0.1, 1.0),
                "num_folds": len(group),
            })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    
    high_conf = sum(1 for n in notes if n.get("num_folds", 1) >= 2)
    print(f"[fretnet] All folds ensemble: {len(notes)} notes "
          f"(high_conf≥2folds: {high_conf}, total_folds: {len(fold_models)})")

    return {
        "notes": notes,
        "total_count": len(notes),
        "method": "fretnet",
        "num_folds": len(fold_models),
    }


if __name__ == "__main__":
    if not is_model_available():
        print("FretNet model not found! Run train_fretnet.py first.")
        sys.exit(1)

    import sys
    wav = sys.argv[1] if len(sys.argv) > 1 else None
    if not wav:
        print("Usage: python fretnet_transcriber.py <wav_path>")
        sys.exit(1)

    result = transcribe_guitar(wav)
    print(f"\nTotal notes: {result['total_count']}")
    for n in result['notes'][:20]:
        print(f"  t={n['start']:.2f}-{n['end']:.2f} s{n['string']} f{n['fret']} MIDI={n['pitch']}")
