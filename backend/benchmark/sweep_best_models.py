"""onset_threshold微調整: 0.7-0.9を0.05刻みでスイープ（ドメイン別ベスト選択モデル）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "music-transcription", "python"))

# benchmark_e2eは使わず直接実装
import argparse

# benchmark_e2eはpure_moe_transcriberのデフォルトパラメータを使う
# onset_thresholdを変えるにはpure_moe_transcriberの引数を直接制御する必要がある
# 代わりにgrid_search_3ds.pyのアプローチでキャッシュベースで高速スイープする

from grid_search_3ds import load_jams_notes, to_mireval, decode_with_params, ANNOTATIONS_DIR, AUDIO_DIR
import glob, numpy as np, torch, librosa, mir_eval
import config
from model import architecture

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "music-transcription", "python", "_processed_guitarset_data", "training_output")

def get_best_models():
    """ドメイン別Best F1モデルを返す"""
    domain_names = [
        "martin_finger", "taylor_finger", "luthier_finger",
        "martin_pick", "taylor_pick", "luthier_pick",
        "gibson_thumb",
    ]
    stage_suffixes = ["multitask_3ds", "multitask", "guitarset_ft"]
    models = []
    for dname in domain_names:
        best_f1 = -1.0
        best = None
        for suffix in stage_suffixes:
            candidate = f"finetuned_{dname}_{suffix}"
            log_path = os.path.join(OUTPUT_DIR, candidate, "training_log.txt")
            model_path = os.path.join(OUTPUT_DIR, candidate, "best_model.pth")
            if not os.path.exists(model_path):
                continue
            f1 = 0.0
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as lf:
                    for line in lf:
                        if "Best F1:" in line:
                            try: f1 = float(line.split("Best F1:")[1].strip().split()[0])
                            except: pass
            if f1 > best_f1:
                best_f1 = f1
                best = candidate
        if best:
            models.append(best)
            print(f"  {dname}: {best} (F1={best_f1:.4f})")
    return models

def run_inference_best(wav_path, models):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    all_onset_probs = []
    all_fret_preds = []
    for model_dir in models:
        model_path = os.path.join(OUTPUT_DIR, model_dir, "best_model.pth")
        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=1280, rnn_type="GRU",
            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        del model, state_dict
        torch.cuda.empty_cache()
    return np.array(all_onset_probs), np.array(all_fret_preds)

if __name__ == "__main__":
    print("=== Best-per-domain + onset sweep ===")
    models = get_best_models()
    print(f"\nModels: {len(models)}")

    # benchmark_e2eと同じ10曲を使用
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    pairs = []
    for jp in jams_files:
        base = os.path.basename(jp).replace(".jams", "")
        wav = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        if os.path.exists(wav):
            pairs.append((jp, wav, base))
    # benchmark_e2e uses first 10 from each player
    from itertools import islice
    selected = pairs[:10]  # first 10

    # Cache inference
    song_data = []
    for i, (jp, wp, name) in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {name}")
        gt = load_jams_notes(jp)
        if not gt: continue
        ri, rp = to_mireval(gt)
        aop, afp = run_inference_best(wp, models)
        song_data.append({"name": name, "ref_intervals": ri, "ref_pitches": rp,
                          "all_onset_probs": aop, "all_fret_preds": afp, "n_gt": len(gt)})

    # Sweep
    print(f"\n{'='*50}")
    for ot in [0.7, 0.75, 0.8, 0.85, 0.9]:
        for vt in [4, 5, 6]:
            for vpt in [0.4, 0.5]:
                total_tp, total_fp, total_fn = 0, 0, 0
                for sd in song_data:
                    pred = decode_with_params(sd["all_onset_probs"], sd["all_fret_preds"], vt, ot, vpt)
                    if not pred:
                        total_fn += sd["n_gt"]
                        continue
                    ei, ep = to_mireval(pred)
                    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                        sd["ref_intervals"], sd["ref_pitches"], ei, ep,
                        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
                    tp = int(round(r * len(sd["ref_intervals"])))
                    fp = len(pred) - int(round(p * len(pred)))
                    fn = len(sd["ref_intervals"]) - tp
                    total_tp += tp; total_fp += max(0,fp); total_fn += max(0,fn)
                mp = total_tp / max(total_tp + total_fp, 1)
                mr = total_tp / max(total_tp + total_fn, 1)
                mf = 2*mp*mr / max(mp+mr, 1e-8)
                marker = " <<<<" if mf > 0.832 else ""
                print(f"  v={vt} o={ot:.2f} vp={vpt:.1f} -> P={mp:.4f} R={mr:.4f} F1={mf:.4f}{marker}")
