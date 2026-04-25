from __future__ import annotations
"""
ensemble_transcriber.py — 複数モデルによる合議制転写 (Ensemble Majority Vote)
=======================================================
FretNet / SynthTab / CRNN / Piano(ByteDance) の4モデルの結果を統合し、
多数決投票で高信頼度のノートリストを生成する。

戦略:
  1. 各モデルを独立に実行
  2. ノートを時間窓(±tolerance)内で照合
  3. 2モデル以上で一致したノートを高信頼度として採用
  4. 1モデルのみのノートも低信頼度で保持

モデル:
  - FretNet: 連続ピッチ推定 (GuitarSet 30分で学習)
  - SynthTab: TabCNN (合成6,700時間で学習)
  - CRNN: CNN+BiGRU (GuitarSetで学習)
  - Piano DA: ピアノ転写 (MAESTRO 200+時間) → ギター音域フィルタ
"""

import time
import numpy as np
import librosa
import json
from typing import List, Dict, Optional, Callable, Set, cast
from pathlib import Path


# --- 音声前処理 (Domain Adaptation) ---

def preprocess_audio_for_transcription(wav_path: str, output_path: Optional[str] = None) -> str:
    """
    YouTube音源をGuitarSet-likeな音質に近づける前処理。
    
    1. ラウドネス正規化 (peak normalize)
    2. ハイパスフィルタ (100Hz以下カット — 低域ノイズ除去)
    3. EQ補正 (低音域-6dB, メロディ帯域+3dB — メロディ検出改善)
    4. ジェントル圧縮 (ダイナミックレンジ縮小)
    """
    import librosa
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    
    if output_path is None:
        output_path = wav_path  # in-place
    
    audio, sr = librosa.load(wav_path, sr=22050, mono=True)
    
    # 1. Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    
    # 2. High-pass filter at 65Hz (ギターの6弦E2=82Hzの基本波を確実に通す、B1以下のノイズのみ除去)
    sos = butter(4, 65, btype='highpass', fs=22050, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)
    
    # 3. EQ補正 — メロディ帯域を強調、低音域を軽く抑制
    # アコギのメロディ帯域(1-4kHz)をブーストし、ベース倍音帯域(100-400Hz)を軽く抑制
    # Low shelf: 250Hz以下を -1.5dB (0.85倍) — ベース音の基本波(82-250Hz)を保全しつつノイズ低減
    sos_low = butter(2, 250, btype='lowpass', fs=22050, output='sos')
    low_band = sosfilt(sos_low, audio).astype(np.float32)
    audio = audio - low_band * 0.15  # 低音域を15%カット (-1.5dB) ← 29%から緩和
    
    # High shelf: 1kHz以上を +2dB (1.26倍)
    sos_high = butter(2, 1000, btype='highpass', fs=22050, output='sos')
    high_band = sosfilt(sos_high, audio).astype(np.float32)
    audio = audio + high_band * 0.25  # 高音域を25%ブースト (+2dB) ← 41%から緩和
    
    # Re-normalize after EQ
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    
    # 4. Gentle compression (reduce dynamic range → helps model detect quiet notes)
    threshold = 0.3
    ratio = 3.0
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    compressed = audio.copy()
    compressed[mask] = np.sign(audio[mask]) * (
        threshold + (abs_audio[mask] - threshold) / ratio
    )
    audio = compressed
    
    # Re-normalize after compression
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    
    sf.write(output_path, audio, 22050)
    return output_path


# --- ノート照合・統合 ---

def _notes_overlap(note_a: Dict, note_b: Dict, 
                   time_tolerance: float = 0.06,
                   pitch_tolerance: int = 1) -> bool:
    """2つのノートが同一と見なせるか判定。"""
    # ピッチの一致（±1半音まで許容）
    pitch_diff = abs(note_a["pitch"] - note_b["pitch"])
    if pitch_diff > pitch_tolerance:
        return False

    # 低音域(E2=40付近)はオンセットの検出時間がモデル間でズレやすいため、
    # 窓幅を広げる。
    effective_tolerance = time_tolerance
    if note_a["pitch"] <= 41:  # E2付近にターゲットを絞る
        effective_tolerance = max(effective_tolerance, 0.3)

    # 時間的な近接度
    onset_diff = abs(note_a["start"] - note_b["start"])
    if onset_diff > effective_tolerance:
        return False
    
    return True


def _merge_note_group(notes: List[Dict]) -> Dict:
    """同一と判定されたノート群を統合して1つのノートにする。"""
    # 優先モデル（高精度な順）
    trusted_sources = ["crnn", "fretnet", "synthtab", "synthtab_egdb", "basic_pitch"]
    
    ref_note_for_time = notes[0]
    best_priority = 999
    
    for n in notes:
        src = n.get("source", "unknown")
        try:
            priority = trusted_sources.index(src)
        except ValueError:
            priority = 999
        if priority < best_priority:
            best_priority = priority
            ref_note_for_time = n
            
    # 平均的なタイミングを使用 (わずかなタイミング差異を残すことでアルペジオを維持)
    avg_start = np.mean([n["start"] for n in notes])
    avg_end = np.mean([n["end"] for n in notes])
    
    # ピッチは最頻値を使用
    from collections import Counter
    pitch_counts = Counter(n["pitch"] for n in notes)
    best_pitch = pitch_counts.most_common(1)[0][0]
    
    # 弦/フレット: 高精度モデルのものを優先（ピッチ合致条件）
    matching_notes = [n for n in notes if n["pitch"] == best_pitch]
    if not matching_notes:
        matching_notes = notes
        
    ref_note = matching_notes[0]
    best_prio_fret = 999
    for mn in matching_notes:
        src = mn.get("source", "unknown")
        try:
            p = trusted_sources.index(src)
        except ValueError:
            p = 999
        if p < best_prio_fret:
            best_prio_fret = p
            ref_note = mn
    
    # 信頼度 = 一致モデル数 / 使用モデル総数
    sources = set()
    for n in notes:
        sources.add(n.get("source", "unknown"))
    
    # CRNN単独検出の場合も、その圧倒的精度から「2モデル相当」として扱いフィルタ残存させる
    effective_votes = len(notes)
    if effective_votes == 1 and "crnn" in sources:
        effective_votes = 2
        
    confidence = effective_votes / 5.0  # 5モデル体制
    
    # velocity: 後続の合意用シグナルとして機能させる (KI準拠)
    base_velocity = min(0.5 + confidence * 0.3, 1.0)
    
    return {
        "start": round(avg_start, 4),
        "end": round(avg_end, 4),
        "pitch": best_pitch,
        "string": ref_note["string"],
        "fret": ref_note["fret"],
        "velocity": min(base_velocity, 1.0),
        "confidence": float(int(float(confidence) * 100 + 0.5)) / 100.0,
        "sources": list(sources),
        "num_models": len(notes),
    }


def ensemble_notes(
    results_list: List[List[Dict]],
    model_names: List[str],
    time_tolerance: float = 0.06,
    pitch_tolerance: int = 1,
    min_score: float = 1.0,
    model_weights: Optional[Dict[str, float]] = None,
    report: Optional[Callable] = None,
) -> List[Dict]:
    """
    複数モデルの出力を統合する。
    
    Parameters
    ----------
    results_list : list[list[dict]]
        各モデルの出力ノートリスト
    model_names : list[str]
        各モデルの名前のリスト
    time_tolerance : float
        同一ノートと見なす時間差の上限（秒）
    pitch_tolerance : int
        同一ノートと見なすピッチ差の上限（半音）
    min_score : float
        採用するのに必要な最小スコア（モデルの重みの合計）
    model_weights : dict[str, float], optional
        各モデルの重みを定義する辞書。デフォルトは全て1.0
    
    Returns
    -------
    list[dict]
        統合されたノートリスト（信頼度付き）
    """
    if model_weights is None:
        model_weights = {name: 1.0 for name in model_names}
    if not results_list:
        return []
    
    # フィルタリング済みの全ノートをフラットに
    all_notes = []
    for model_idx, notes in enumerate(results_list):
        for note in notes:
            note_copy = dict(note)
            note_copy["_model_idx"] = model_idx
            all_notes.append(note_copy)
    
    if not all_notes:
        return []
    
    # 時間順にソート
    all_notes.sort(key=lambda n: (n["start"], n["pitch"]))
    
    # グリーディマッチング: 各ノートをグループにまとめる
    used = [False] * len(all_notes)
    groups = []
    
    for i in range(len(all_notes)):
        if used[i]:
            continue
        
        group = [all_notes[i]]
        used[i] = True
        model_indices_in_group = {all_notes[i]["_model_idx"]}
        
        # このノートと一致する他のノートを探す
        for j in range(i + 1, len(all_notes)):
            if used[j]:
                continue
            
            # 低音域の緩和対応
            eff_tol = time_tolerance
            if all_notes[i]["pitch"] <= 41:
                eff_tol = max(eff_tol, 0.3)
            
            # 時間的に遠すぎたら打ち切り
            if all_notes[j]["start"] - all_notes[i]["start"] > eff_tol * 1.2:
                break
            
            # 同じモデルからの重複は避ける
            if all_notes[j]["_model_idx"] in model_indices_in_group:
                continue
            
            if _notes_overlap(all_notes[i], all_notes[j], 
                            time_tolerance, pitch_tolerance):
                group.append(all_notes[j])
                used[j] = True
                model_indices_in_group.add(all_notes[j]["_model_idx"])
        
        groups.append(group)
    
    # グループを統合ノートに変換
    merged_notes = []
    for group in groups:
        # B2 (pitch 47) 以下のベース音を含むグループは特別扱い
        is_low_base = any(n["pitch"] < 48 for n in group)
        
        # モデルごとの重みの合計を計算
        group_score = sum(model_weights.get(model_names[n["_model_idx"]], 1.0) for n in group)
        models_in_group = [model_names[n["_model_idx"]] for n in group]
        
        # 評価: ベース音は無条件または低閾値で、他は min_score に達するか
        if is_low_base:
            if group_score < min_score:
                if report is not None:
                    cast(Callable, report)(f"E2 Group ACCEPTED by relaxation: score={group_score:.1f}/{min_score:.1f}, models={models_in_group}")
            else:
                if report is not None:
                    cast(Callable, report)(f"E2 Group Found: score={group_score:.1f}, models={models_in_group}, starts={[round(float(n['start']), 2) for n in group]}")
            
            merged = _merge_note_group(group)
            merged["num_models"] = max(1, int(group_score)) # 互換性のため
            merged_notes.append(merged)
        elif group_score >= min_score:
            merged = _merge_note_group(group)
            merged_notes.append(merged)
merged_notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return merged_notes


def _filter_notes(notes: List[Dict], report=None) -> List[Dict]:
    """
    ノートフィルタリング — アンサンブル統合直後の最小限フィルタ。
    
    詳細なフィルタリング (velocity, duration, harmonics) は
    pipeline.py の apply_all_filters で一元的に行うため、
    ここでは重複除去と弦数制限のみ実施。
    """
    if not notes:
        return notes
    
    def log(msg):
        if report:
            report(msg)
    
    initial_count = len(notes)
    
    # 時間順にソート
    notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))
    
    # Step 1: 完全重複ノート除去 (同時刻±20ms, 同ピッチ)
    deduped = []
    for n in notes:
        is_dup = False
        for existing in deduped[-10:]:
            if (abs(n["start"] - existing["start"]) < 0.02 and
                n["pitch"] == existing["pitch"]):
                is_dup = True
                if n.get("end", 0) > existing.get("end", 0):
                    existing["end"] = n["end"]
                if n.get("num_models", 1) > existing.get("num_models", 1):
                    existing["num_models"] = n.get("num_models", 1)
                    existing["confidence"] = n.get("confidence", "low")
                break
        if not is_dup:
            deduped.append(n)
    notes = deduped
    
    # Step 2: 同時発音数制限（最大6 = ギター弦数）
    filtered = []
    i = 0
    while i < len(notes):
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j]["start"] - notes[i]["start"] < 0.015:
            group.append(notes[j])
            j += 1
        
        if len(group) > 6:
            group.sort(key=lambda n: (
                -n.get("num_models", 1),
                -n.get("velocity", 0.5),
            ))
            group = group[:6]
        
        filtered.extend(group)
        i = j
    notes = filtered
    
    notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))
    
    removed = initial_count - len(notes)
    log(f"フィルタ後: {len(notes)} notes (除去: {removed})")
    
    return notes


def transcribe_ensemble(
    wav_path: str,
    tuning_pitches: Optional[Dict[int, int]] = None,
    progress_cb: Optional[Callable] = None,
    min_models_for_accept: int = 1,
    output_path: Optional[str] = None,
    report: Optional[Callable] = None,
) -> Dict:
    """
    6つのMoEエキスパートモデルを順次ロードし、Frame-Level Hard Consensus (多数決) で統合する。
    """
    report_lines = []
    def _report(msg):
        report_lines.append(msg)
        if progress_cb:
            progress_cb("notes", msg)
        if report:
            report(msg)
        print(f"[ensemble] {msg}")

    import torch
    import numpy as np
    import time
    from model import architecture
    import config
    import librosa
    from scipy import stats
    from guitar_transcriber import _frames_to_notes
    from pathlib import Path
    import os
    import sys
    import json
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _report(f"Starting Hard Consensus MoE Ensemble (6 models) on {device}...")
    
    t0 = time.time()
    _report("Extracting CQT features...")
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    all_onset_probs = []
    all_fret_preds = []
    
    models_to_test = [
        "finetuned_martin_finger_model",
        "finetuned_taylor_finger_model",
        "finetuned_luthier_finger_model",
        "finetuned_martin_pick_model",
        "finetuned_taylor_pick_model",
        "finetuned_luthier_pick_model"
    ]
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    mt_python_dir = os.path.join(project_root, "..", "music-transcription", "python")
    if mt_python_dir not in sys.path:
        sys.path.insert(0, mt_python_dir)
    
    loaded_count = 0
    for model_dir in models_to_test:
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            _report(f"Skipping {model_dir}: Not found at {model_path}")
            continue
            
        _report(f"Inferencing with Expert {loaded_count+1}/6: {model_dir}...")
        model = architecture.GuitarTabCRNN(num_frames_rnn_input_dim=1280, rnn_type="GRU", rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
            
        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        loaded_count += 1
        
        # OOM回避のため即座にVRAMから破棄
        del model
        del state_dict
        torch.cuda.empty_cache()
        
    if loaded_count == 0:
        raise RuntimeError("No MoE models found for Hard Consensus. Please train the expert models.")
        
    _report(f"Voting on Frame-Level probabilities (threshold: 3/{loaded_count} votes)...")
    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)
    
    # Hard Consensus Voting
    vote_threshold = 3
    binary_votes = all_onset_probs > 0.4
    vote_counts = np.sum(binary_votes, axis=0) # [Frames, Strings]
    
    # 全モデルの平均(np.mean)をとると、確信度の低いモデルに足を引っ張られて全体が下がり、
    # 閾値(0.65)を割って消滅してしまう。そのため最大値(np.max)を採用し、強いシグナルを保つ。
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < vote_threshold] = 0.0
    
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)
    
    _report("Decoding to symbolic notes...")
    notes = _frames_to_notes(
        consensus_onset_probs, consensus_frets, tuning_pitches=tuning_pitches, onset_threshold=0.5
    )
    
    for n in notes:
        n["source"] = "hard_consensus_moe"
        n["confidence"] = n.get("velocity", 0.5)
        n["num_models"] = vote_threshold
        
    elapsed = time.time() - t0
    _report(f"Hard Consensus Complete: {len(notes)} notes ({elapsed:.1f}s)")
    
    # フィルタ・テクニック付与等
    notes = _filter_notes(notes, _report)
    notes = _annotate_techniques(wav_path, notes, _report)
    notes = _annotate_string_hints(wav_path, notes, _report)
    
    if output_path:
        try:
            report_path = Path(output_path).parent / "ensemble_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
        except Exception as re:
            _report(f"レポート保存失敗: {re}")

    return {
        "notes": notes,
        "total_count": len(notes),
        "method": "HardConsensusMoE",
        "model_stats": {
            "ensemble": {
                "total_notes": len(notes),
                "models_used": models_to_test,
                "num_models": loaded_count
            }
        },
    }


def _annotate_techniques(wav_path: str, notes: List[Dict], report=None) -> List[Dict]:
    """
    テクニック分類器でノートにtechnique属性を付与。
    technique: normal / palm_mute / harmonic
    """
    model_path = Path(__file__).parent.parent / "generated" / "technique_classifier" / "best_model.pt"
    if not model_path.exists():
        if report:
            report("テクニック分類: モデル未検出、スキップ")
        return notes
    
    try:
        import torch
        from technique_classifier import TechniqueClassifierCNN, TECHNIQUE_CLASSES
        
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model = TechniqueClassifierCNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load audio once
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        
        tech_counts = {}
        seg_dur = 0.3
        seg_samples = int(seg_dur * 22050)
        
        for note in notes:
            onset = note.get('start', 0)
            start_sample = max(0, int(onset * 22050) - int(0.02 * 22050))
            end_sample = start_sample + seg_samples
            
            if end_sample > len(y):
                note['technique'] = 'normal'
                continue
            
            segment = y[start_sample:end_sample]
            if len(segment) < seg_samples:
                segment = np.pad(segment, (0, seg_samples - len(segment)))
            
            mel = librosa.feature.melspectrogram(
                y=segment.astype(np.float32), sr=22050,
                n_mels=64, n_fft=1024, hop_length=256
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
            
            x = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]
                pred = probs.argmax().item()
                confidence = probs[pred].item()
            
            tech_name = TECHNIQUE_CLASSES[pred] if pred < len(TECHNIQUE_CLASSES) else 'normal'
            # 信頼度閾値: normal以外は高確信(>0.95)の場合のみ採用
            if tech_name != 'normal' and confidence < 0.95:
                tech_name = 'normal'
            note['technique'] = tech_name
            note['technique_confidence'] = round(confidence, 3)
            tech_counts[tech_name] = tech_counts.get(tech_name, 0) + 1
        
        tech_str = ', '.join(f'{k}:{v}' for k, v in sorted(tech_counts.items()))
        if report:
            report(f"テクニック推定完了: {tech_str}")
    
    except Exception as e:
        if report:
            report(f"テクニック推定エラー: {e}")
        for note in notes:
            note['technique'] = 'normal'
    
    return notes


def _annotate_string_hints(wav_path: str, notes: List[Dict], report=None) -> List[Dict]:
    """
    弦推定CNNでノートにcnn_string_probs属性を付与。
    cnn_string_probs: [6弦prob, 5弦prob, ..., 1弦prob] (string_assigner用)
    """
    model_path = Path(__file__).parent.parent / "generated" / "string_classifier" / "best_model.pt"
    if not model_path.exists():
        if report:
            report("弦推定CNN: モデル未検出、スキップ")
        return notes
    
    try:
        import torch
        import torch.nn.functional as F
        from string_classifier import StringClassifierCNN, NUM_STRINGS
        
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model = StringClassifierCNN(NUM_STRINGS)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        
        seg_dur = 0.3
        seg_samples = int(seg_dur * 22050)
        annotated = 0
        
        for note in notes:
            onset = note.get('start', 0)
            midi_pitch = note.get('pitch', 60)
            start_sample = max(0, int(onset * 22050) - int(0.02 * 22050))
            end_sample = start_sample + seg_samples
            
            if end_sample > len(y):
                continue
            
            segment = y[start_sample:end_sample]
            if len(segment) < seg_samples:
                segment = np.pad(segment, (0, seg_samples - len(segment)))
            
            mel = librosa.feature.melspectrogram(
                y=segment.astype(np.float32), sr=22050,
                n_mels=64, n_fft=1024, hop_length=256
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
            
            x = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
            pitch_norm = (midi_pitch - 30.0) / 60.0
            pitch_tensor = torch.FloatTensor([[pitch_norm]])
            
            with torch.no_grad():
                out = model(x, pitch_tensor)
                probs = F.softmax(out, dim=1)[0].numpy()
            
            # probs[0]=6弦(E2), probs[5]=1弦(E4) — string_assigner uses 1-6
            # Convert: string_num 6 = probs[0], string_num 1 = probs[5]
            note['cnn_string_probs'] = {
                6: float(probs[0]), 5: float(probs[1]), 4: float(probs[2]),
                3: float(probs[3]), 2: float(probs[4]), 1: float(probs[5]),
            }
            annotated += 1
        
        if report:
            report(f"弦推定CNN: {annotated}/{len(notes)} ノートにヒント付与")
    
    except Exception as e:
        if report:
            report(f"弦推定CNNエラー: {e}")
    
    return notes


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ensemble_transcriber.py <wav_path>")
        sys.exit(1)
    
    wav = sys.argv[1]
    print(f"=== Ensemble Transcription: {wav} ===")
    
    # 前処理
    preprocessed = wav.replace(".wav", "_preprocessed.wav")
    preprocess_audio_for_transcription(wav, preprocessed)
    print(f"Preprocessed: {preprocessed}")
    
    result = transcribe_ensemble(preprocessed)
    print(f"\n=== Results ===")
    print(f"Total notes: {result['total_count']}")
    print(f"Models: {result['model_stats']}")
    
    # 信頼度分布
    from collections import Counter
    conf_dist = Counter(n.get("num_models", 1) for n in result["notes"])
    print(f"Confidence distribution: {dict(conf_dist)}")
    
    print(f"\nFirst 20 notes:")
    for n in result["notes"][:20]:
        src = n.get("sources", ["?"])
        print(f"  t={n['start']:.2f}-{n['end']:.2f} s{n['string']} f{n['fret']} "
              f"MIDI={n['pitch']} conf={n.get('confidence', '?')} src={src}")

def transcribe_domain_ensemble(
    wav_path: str,
    tuning_pitches: Optional[Dict[int, int]] = None,
    progress_cb: Optional[Callable] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    6つの各ドメイン専用モデル (martin/taylor/luthier x finger/pick) をロードして推論し、
    結果を「Domain-Ensemble」として多数決統合する。
    自動的にVRAMを解放・節約する設計。
    """
    report_lines = []
    def report(msg):
        report_lines.append(msg)
        if progress_cb:
            progress_cb("notes", msg)
        print(f"[domain-ensemble] {msg}")
    
    results = []
    model_names = []
    model_stats = {}
    
    domains = [
        "martin_finger", "taylor_finger", "luthier_finger", 
        "martin_pick", "taylor_pick", "luthier_pick"
    ]
    
    import guitar_transcriber as gt
    import torch
    import json
    
    report("Starting Domain-Ensemble sequential transcription...")
    
    base_out_dir = gt._MT_DIR / "_processed_guitarset_data" / "training_output"
    
    for domain in domains:
        model_name = f"finetuned_{domain}_model"
        target_dir = base_out_dir / model_name
        model_path = target_dir / "best_model.pth"
        config_path = target_dir / "run_configuration.json"
        
        if not model_path.exists() or not config_path.exists():
            report(f"Skipping {domain}: Model or config not found.")
            continue
            
        report(f"Loading Specialized Model: {domain}")
        
        original_get_paths = gt._get_model_paths
        # 一時的にパスをモックして guitar_transcriber にロードさせる
        gt._get_model_paths = lambda: (str(model_path), str(config_path))
        gt._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            t0 = time.time()
            crnn_result = gt.transcribe_guitar(wav_path, tuning_pitches=tuning_pitches)
            crnn_notes = crnn_result["notes"]
            
            for n in crnn_notes:
                n["source"] = domain
            results.append(crnn_notes)
            model_names.append(domain)
            elapsed = time.time() - t0
            model_stats[domain] = {"notes": len(crnn_notes), "time": round(float(elapsed), 1)}
            report(f"Domain model {domain}: {len(crnn_notes)} notes ({elapsed:.1f}s)")
                
        except Exception as e:
            report(f"Error inferring with {domain}: {e}")
        finally:
            gt._get_model_paths = original_get_paths
            gt._model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    num_models = len(results)
    if num_models == 0:
        raise RuntimeError("利用可能なドメインモデルがありません")
        
    report(f"Domain-Ensemble統合中 ({num_models}モデル: {', '.join(model_names)})...")
        
    model_weights = {name: 1.0 for name in model_names}
    
    import math
    new_min_score = math.floor(num_models / 2.0)
    if new_min_score < 1: new_min_score = 1
    if num_models == 6: new_min_score = 3
    
    notes = ensemble_notes(
        results,
        model_names=model_names,
        time_tolerance=0.08,
        pitch_tolerance=0,    
        min_score=new_min_score,
        model_weights=model_weights,
        report=report,
    )
    
    before_filter = len(notes)
    notes = _filter_notes(notes, report)
    report(f"フィルタ後: {len(notes)} notes (除去: {before_filter - len(notes)})")
    
    notes = _annotate_techniques(wav_path, notes, report)
    notes = _annotate_string_hints(wav_path, notes, report)
    
    model_stats["ensemble"] = {
        "total_notes": len(notes),
        "models_used": model_names,
        "num_models": num_models,
        "min_score_applied": new_min_score
    }
    
    if output_path:
        report_path = Path(output_path).parent / "domain_ensemble_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
    return {
        "notes": notes,
        "total_count": len(notes),
        "method": "domain_ensemble",
        "model_stats": model_stats,
    }
