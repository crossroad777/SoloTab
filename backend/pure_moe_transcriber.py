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

def transcribe_pure_moe(wav_path: str, vote_threshold: int = 5, onset_threshold: float = 0.75, vote_prob_threshold: float = 0.5) -> list:
    """
    指定された7つの特化モデルのみを使って純粋に音符を検出するトランスクライバ。
    他のフィルタリングや後処理（DP等）は一切行わず、モデルの出力を忠実に返す。

    最適パラメータ (3DS + ドメイン別Best F1選択, 2026-05-05):
      vote_threshold=5, onset_threshold=0.75, vote_prob_threshold=0.5
      → E2E 10曲: F1=0.8323 (P=0.8543, R=0.8114)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting CQT features for {os.path.basename(wav_path)} ...")
    
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 各ドメインの最良モデルを動的に選択（最新ステージ優先）
    domain_names = [
        "martin_finger", "taylor_finger", "luthier_finger",
        "martin_pick", "taylor_pick", "luthier_pick",
        "gibson_thumb",
    ]
    # Best F1で動的選択: _3ds (個別F1=0.78-0.80) > _3ds_ga (0.75-0.78) > _3ds_ga_gc
    stage_suffixes = ["multitask_3ds", "multitask_3ds_ga", "multitask_3ds_ga_gc"]
    
    models_to_test = []
    for dname in domain_names:
        best_f1 = -1.0
        best_candidate = None
        for suffix in stage_suffixes:
            candidate = f"finetuned_{dname}_{suffix}"
            candidate_dir = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", candidate)
            candidate_path = os.path.join(candidate_dir, "best_model.pth")
            if not os.path.exists(candidate_path):
                continue
            # training_logからBest F1を読み取り
            log_path = os.path.join(candidate_dir, "training_log.txt")
            f1 = 0.0
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as lf:
                    for line in lf:
                        if "Best F1:" in line:
                            try:
                                f1 = float(line.split("Best F1:")[1].strip().split()[0])
                            except:
                                pass
            if f1 > best_f1:
                best_f1 = f1
                best_candidate = candidate
            elif best_candidate is None:
                best_candidate = candidate
        if best_candidate:
            models_to_test.append(best_candidate)
    # --- モデル一括ロード（キャッシュ利用） ---
    global _CACHED_MODELS
    if '_CACHED_MODELS' not in globals():
        _CACHED_MODELS = {}
    
    models_loaded = []
    for model_dir in models_to_test:
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            continue
        
        if model_dir in _CACHED_MODELS:
            models_loaded.append((model_dir, _CACHED_MODELS[model_dir]))
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
    
    print(f"Models ready: {len(models_loaded)} (cached: {len(_CACHED_MODELS)})")
    
    # --- 一括推論 ---
    all_onset_probs = []
    all_fret_preds = []
    
    with torch.no_grad():
        for i, (name, model) in enumerate(models_loaded):
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
            all_onset_probs.append(onset_probs)
            all_fret_preds.append(np.argmax(fret_probs, axis=-1))
            print(f"  Expert {i+1}/{len(models_loaded)}: {name} done")
        
    if not all_onset_probs:
        print("No models were successfully loaded.")
        return []
        
    print(f"Voting on frame-level predictions (threshold: {vote_threshold}/{len(all_onset_probs)})...")
    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)
    
    # 多数決ロジック
    # vote_prob_threshold以上の確率を出したモデルの数を数える
    binary_votes = all_onset_probs > vote_prob_threshold
    vote_counts = np.sum(binary_votes, axis=0) # [Frames, Strings]
    
    # ピークを維持するため、平均ではなく最大の確率を採用
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    # 閾値未満の合意フレームをカット
    consensus_onset_probs[vote_counts < vote_threshold] = 0.0
    
    # フレットは最頻値をそのまま採用（DP等を使わない）
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)
    
    print("Decoding probabilities to notes...")
    notes = _frames_to_notes(
        consensus_onset_probs, consensus_frets, tuning_pitches=None, onset_threshold=onset_threshold
    )
    
    for n in notes:
        n["source"] = "pure_moe"
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])
    
    # --- 物理制約フィルタ: 一時無効化（ベースライン比較用） ---
    # before_count = len(notes)
    # notes = _apply_physical_constraints(notes)
    # removed = before_count - len(notes)
    # if removed > 0:
    #     print(f"Physical filter: removed {removed} notes ({before_count} -> {len(notes)})")
        
    print(f"Completed! Detected {len(notes)} notes.")
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
