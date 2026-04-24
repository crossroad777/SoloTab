import torch
import os
import sys
import numpy as np
from typing import List, Dict, Callable
from solotab_utils import _to_native

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

# 推論時にロードして捨てるためのダミーアーキテクチャインポート
from model import architecture

EXPERTS_PATH = {
    "martin_finger": r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\finetuned_martin_finger_model\best_model.pth",
    "taylor_finger": r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\finetuned_taylor_finger_model\best_model.pth",
    "luthier_finger": r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\finetuned_luthier_finger_model\best_model.pth",
    "martin_pick": r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\finetuned_martin_pick_model\best_model.pth",
    "taylor_pick": r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\finetuned_taylor_pick_model\best_model.pth",
    "luthier_pick": r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\finetuned_luthier_pick_model\best_model.pth"
}

def load_router(device):
    """Gating Networkのロード"""
    from moe_gating_network import GatingNetworkCNN
    model = GatingNetworkCNN(in_channels=1, num_experts=6).to(device)
    router_path = os.path.join(project_root, "music-transcription", "python", "_processed_guitarset_data", "training_output", "finetuned_router_model", "latest_model.pth")
    if os.path.exists(router_path):
        model.load_state_dict(torch.load(router_path, map_location=device, weights_only=True))
    else:
        print(f"[Warning] Router model not found at {router_path}")
    model.eval()
    return model

def extract_features(wav_path: str):
    """全モデル共通で使うCQT特徴量を一度だけ計算してメモリに保持"""
    import librosa
    import config
    import torchaudio
    
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return features

def transcribe_dynamic_ensemble(wav_path: str, tuning_pitches: Dict = None, progress_cb: Callable = None, output_path: str = None, onset_threshold: float = 0.5, return_raw_logits: bool = False, **kwargs) -> Dict:
    """
    案Dに基づく動的MoE統合推論エンジン
    OOMを防ぐため、1つずつモデルをロード・推論し、即座にメモリから破棄する（シーケンシャルロード）。
    """
    def report(msg):
        if progress_cb:
            progress_cb("notes", msg)
        print(f"[Ensemble] {msg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report(f"Starts dynamic MoE inference on {os.path.basename(wav_path)}")
    
    # 1. 共通特徴量抽出
    report("Extracting CQT features...")
    features = extract_features(wav_path).to(device)
    
    # 2. Routerによる重み付け計算
    report("Loading MoE Router...")
    router = load_router(device)
    with torch.no_grad():
        # gating_weights is a tensor of shape (1, 6)
        gating_weights = router(features)
        weights_array = gating_weights.cpu().numpy()[0]
        
    del router
    torch.cuda.empty_cache()
    
    # アロケーション（累積確率）
    # shapeは全フレーム長 × ピッチ数 (例: onset_probs, fret_probs)
    accumulated_onset_logits = None
    accumulated_fret_logits = None
    
    # 3. 6モデルのシーケンシャル推論
    expert_names = list(EXPERTS_PATH.keys())
    for i, exp_name in enumerate(expert_names):
        weight = float(weights_array[i])
        if weight < 0.001:
            report(f"Skipping {exp_name} (Weight: {weight:.4f})")
            continue

        report(f"Processing expert {i+1}/6: {exp_name} (Weight: {weight:.4f})")
        
        # モデルロード領域
        model = architecture.GuitarTabCRNN(num_frames_rnn_input_dim=1280, rnn_type="GRU", rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
        state_dict = torch.load(EXPERTS_PATH[exp_name], map_location=device, weights_only=False)
        # unwrap module if dicted from DataParallel
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            # Logits or Probs? We use Softmax/Sigmoid locally to get probabilities
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
            
        # Weighted Addition (Late Fusion)
        if accumulated_onset_logits is None:
            accumulated_onset_logits = onset_probs * weight
            accumulated_fret_logits = fret_probs * weight
        else:
            accumulated_onset_logits += onset_probs * weight
            accumulated_fret_logits += fret_probs * weight
            
        # VRAM解放
        del model
        del state_dict
        torch.cuda.empty_cache()
        # report(f"Finished and unloaded {exp_name}.")
        
    # 4. 確率行列を Note list（辞書の配列）にパースして返す
    report("Decoding accumulated probability matrices into notes...")
    fret_preds_full = np.argmax(accumulated_fret_logits, axis=-1)
    
    from guitar_transcriber import _frames_to_notes
    notes = _frames_to_notes(
        accumulated_onset_logits, 
        fret_preds_full,
        tuning_pitches=tuning_pitches,
        onset_threshold=onset_threshold
    )
    
    report(f"Detected {len(notes)} notes (threshold={onset_threshold:.2f}).")
    
    result = {
        "method": "Dynamic_MoE",
        "total_notes": len(notes),
        "notes": notes,
        "gating_result": {expert_names[i]: float(weights_array[i]) for i in range(6)}
    }
    
    if return_raw_logits:
        result["raw_onset_logits"] = accumulated_onset_logits
        result["raw_fret_logits"] = accumulated_fret_logits

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(_to_native(result), f, ensure_ascii=False, indent=2)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python moe_ensemble_transcriber.py <wav_path>")
        sys.exit(1)
        
    test_wav = sys.argv[1]
    result = transcribe_dynamic_ensemble(test_wav)
    print("\n--- Final Output ---")
    print(f"Total Notes: {result['total_notes']}")
    print("Gating Weights:")
    for k, v in result['gating_result'].items():
        print(f"  {k}: {v:.4f}")
    if result['notes']:
        print("First 10 notes:")
        for n in result['notes'][:10]:
            print(f"  t={n['start']:.2f}-{n['end']:.2f} s{n['string']} f{n['fret']} MIDI={n['pitch']} v={n['velocity']:.2f}")
