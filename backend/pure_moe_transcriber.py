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

def transcribe_pure_moe(wav_path: str, vote_threshold: int = 3, onset_threshold: float = 0.5) -> list:
    """
    指定された6つの特化モデルのみを使って純粋に音符を検出するトランスクライバ。
    他のフィルタリングや後処理（DP等）は一切行わず、モデルの出力を忠実に返す。
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
    
    models_to_test = [
        "finetuned_martin_finger_model",
        "finetuned_taylor_finger_model",
        "finetuned_luthier_finger_model",
        "finetuned_martin_pick_model",
        "finetuned_taylor_pick_model",
        "finetuned_luthier_pick_model"
    ]
    
    all_onset_probs = []
    all_fret_preds = []
    
    for i, model_dir in enumerate(models_to_test):
        print(f"Loading and inferencing Expert {i+1}/{len(models_to_test)}: {model_dir}")
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"  -> Model not found, skipping.")
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
        
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
            
        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        
        del model
        del state_dict
        torch.cuda.empty_cache()
        
    if not all_onset_probs:
        print("No models were successfully loaded.")
        return []
        
    print(f"Voting on frame-level predictions (threshold: {vote_threshold}/{len(all_onset_probs)})...")
    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)
    
    # 多数決ロジック
    # 0.4以上の確率を出したモデルの数を数える
    binary_votes = all_onset_probs > 0.4
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
        # jsonに保存しやすいようにfloatに変換
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])
        
    print(f"Completed! Detected {len(notes)} notes.")
    return notes

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
