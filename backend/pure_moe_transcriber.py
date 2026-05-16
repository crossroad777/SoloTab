import os
import sys
import torch
import numpy as np
import librosa
from scipy import stats

project_root = os.path.dirname(os.path.abspath(__file__))
mt_python_dir = os.path.join(project_root, "..", "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config
from model import architecture
from guitar_transcriber import _frames_to_notes

# 高速モード用ステージ構成 (21モデル: modelと3ds_gaを除外)
_FAST_STAGES = [
    "guitarset_ft",     # + GuitarSet FT
    "multitask",        # + GAPS混合
    "multitask_3ds",    # + GAPS + AG-PT混合
]
# フルモード用ステージ構成 (35モデル)
_FULL_STAGES = [
    "model",            # 合成データのみ（事前学習）
    "guitarset_ft",     # + GuitarSet FT
    "multitask",        # + GAPS混合
    "multitask_3ds",    # + GAPS + AG-PT混合
    "multitask_3ds_ga", # + GAPS + Synth V2混合
]
_DOMAINS = [
    "martin_finger", "taylor_finger", "luthier_finger",
    "martin_pick", "taylor_pick", "luthier_pick",
    "gibson_thumb",
]

# モデルキャッシュ（モジュールレベル — プリロードと推論で共有）
_CACHED_MODELS = {}



def transcribe_pure_moe(wav_path: str, vote_threshold: int = None,
                        onset_threshold: float = 0.75,
                        vote_prob_threshold: float = 0.5,
                        fast_mode: bool = True) -> list:
    """
    MoEトランスクライバ（合議制推論）。

    Parameters
    ----------
    fast_mode : bool (default=True)
        True:  21モデル (7ドメイン × 3ステージ), vote_threshold=13
        False: 35モデル (7ドメイン × 5ステージ), vote_threshold=21

    最適パラメータ (全ステージ統合, 論文 Section 10.15):
      35モデル: vote_threshold=21 → F1=0.8916
      21モデル: vote_threshold=13 → F1≈0.885 (推定)
    """
    import time as _time
    timings = {}

    stage_suffixes = _FAST_STAGES if fast_mode else _FULL_STAGES
    n_expected = len(_DOMAINS) * len(stage_suffixes)
    if vote_threshold is None:
        vote_threshold = max(2, round(n_expected * 0.43))  # 43%合議 (35→15, 21→9, 7→3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_label = f"FAST({len(stage_suffixes)}stg)" if fast_mode else f"FULL({len(stage_suffixes)}stg)"
    print(f"[MoE] {mode_label} on {device}, vote_threshold={vote_threshold}/{n_expected}")

    t_cqt = _time.time()
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    timings['cqt'] = _time.time() - t_cqt
    print(f"[MoE] CQT: {timings['cqt']:.1f}s")
    
    models_to_test = []
    for dname in _DOMAINS:
        for suffix in stage_suffixes:
            candidate = f"finetuned_{dname}_{suffix}"
            candidate_dir = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", candidate)
            candidate_path = os.path.join(candidate_dir, "best_model.pth")
            if os.path.exists(candidate_path):
                models_to_test.append(candidate)
    # --- モデル一括ロード（キャッシュ利用） ---
    global _CACHED_MODELS
    
    t_load = _time.time()
    models_loaded = []
    n_cached = 0
    for model_dir in models_to_test:
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            continue
        
        if model_dir in _CACHED_MODELS:
            models_loaded.append((model_dir, _CACHED_MODELS[model_dir]))
            n_cached += 1
            continue
        
        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=1280, rnn_type="GRU", 
            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _CACHED_MODELS[model_dir] = model
        models_loaded.append((model_dir, model))
    timings['model_load'] = _time.time() - t_load
    n_actual = len(models_loaded)
    print(f"[MoE] Models: {n_actual} loaded ({n_cached} cached, {n_actual-n_cached} new) in {timings['model_load']:.1f}s")
    
    # vote_threshold を実際のモデル数に合わせて再調整
    # (Modal環境では35モデル中7モデルしか存在しない場合がある)
    if n_actual < n_expected and vote_threshold > n_actual:
        old_vt = vote_threshold
        vote_threshold = max(2, round(n_actual * 0.43))
        print(f"[MoE] vote_threshold auto-adjusted: {old_vt} → {vote_threshold} (only {n_actual}/{n_expected} models available)")
    
    # --- 一括推論 ---
    all_onset_probs = []
    all_fret_preds = []
    
    t_infer = _time.time()
    with torch.no_grad():
        for i, (name, model) in enumerate(models_loaded):
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
            all_onset_probs.append(onset_probs)
            all_fret_preds.append(np.argmax(fret_probs, axis=-1))
    timings['inference'] = _time.time() - t_infer
    print(f"[MoE] Inference: {len(models_loaded)} models in {timings['inference']:.1f}s ({timings['inference']/max(len(models_loaded),1):.2f}s/model)")
        
    if not all_onset_probs:
        print("No models were successfully loaded.")
        return []
        
    t_vote = _time.time()
    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)
    
    # 多数決ロジック
    binary_votes = all_onset_probs > vote_prob_threshold
    vote_counts = np.sum(binary_votes, axis=0)
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < vote_threshold] = 0.0
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)
    
    notes = _frames_to_notes(
        consensus_onset_probs, consensus_frets, tuning_pitches=None, onset_threshold=onset_threshold
    )
    timings['voting'] = _time.time() - t_vote
    
    for n in notes:
        n["source"] = "pure_moe"
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])
    
    total = sum(timings.values())
    print(f"[MoE] DONE: {len(notes)} notes in {total:.1f}s "
          f"(CQT={timings['cqt']:.1f}s, Load={timings['model_load']:.1f}s, "
          f"Infer={timings['inference']:.1f}s, Vote={timings['voting']:.1f}s)")
    return notes


def _apply_physical_constraints(notes: list) -> list:
    """
    物理的に不可能なノートを除去する最小限のフィルタ。
    
    除去ルール:
    1. 同一弦で30ms以内に連続する重複ノート（後のものを除去）
    2. 極端に短いノート（< 25ms）
    3. フレット範囲外（> 19フレット）
    """
    if not notes:
        return notes
    
    filtered = []
    MIN_INTERVAL = 0.030   # 同一弦の最小間隔 30ms
    MIN_DURATION = 0.025   # 最小ノート長 25ms
    MAX_FRET = 15
    
    for note in notes:
        # Rule 3: フレット範囲チェック
        if note.get('fret', 0) > MAX_FRET:
            continue
        
        # Rule 2: 最小ノート長チェック
        duration = note.get('end', 0) - note.get('start', 0)
        if duration < MIN_DURATION:
            continue
        
        # Rule 1: 同一弦の重複チェック（直前のノートとの比較）
        duplicate = False
        for prev in reversed(filtered):
            time_gap = note['start'] - prev['start']
            if time_gap > MIN_INTERVAL:
                break
            if prev.get('string') == note.get('string') and prev.get('pitch') == note.get('pitch'):
                duplicate = True
                break
        
        if not duplicate:
            filtered.append(note)
    
    return filtered

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to input wav file")
    parser.add_argument("--vote", type=int, default=3, help="Required number of expert votes")
    parser.add_argument("--thresh", type=float, default=0.5, help="Onset probability threshold")
    parser.add_argument("--out", type=str, default="pure_moe_output.json", help="Output JSON file")
    args = parser.parse_args()
    
    notes = transcribe_pure_moe(args.wav_path, vote_threshold=args.vote, onset_threshold=args.thresh)
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)
    print(f"Saved to {args.out}")
