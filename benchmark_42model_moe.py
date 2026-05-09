"""
benchmark_42model_moe.py — 全42モデル投入MoEベンチマーク
================================================================
全ステージ（model, guitarset_ft, multitask, multitask_3ds, multitask_3ds_ga, multitask_4ds）
× 7ドメインの42モデルを合議させ、vote_threshold 12-30をスイープする。

Usage:
    python benchmark_42model_moe.py
"""
import os, sys, glob, copy, json, time
import numpy as np
import torch
import librosa
from scipy import stats

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
mt_python_dir = os.path.join(r'D:\Music\nextchord-solotab', 'music-transcription', 'python')
sys.path.insert(0, mt_python_dir)

import config
from model import architecture
from guitar_transcriber import _frames_to_notes
from string_assigner import assign_strings_dp, STANDARD_TUNING
import jams

# === Config ===
TRAINING_OUTPUT = os.path.join(mt_python_dir, '_processed_guitarset_data', 'training_output')
ANN_DIR = r'D:\Music\Datasets\GuitarSet\annotation'
MIC_DIR = r'D:\Music\Datasets\GuitarSet\audio_mono-mic'
TEST_PLAYERS = ['05']
ONSET_TOLERANCE = 0.05
ONSET_THRESHOLD = 0.75
VOTE_PROB_THRESHOLD = 0.5

DOMAIN_NAMES = [
    "martin_finger", "taylor_finger", "luthier_finger",
    "martin_pick", "taylor_pick", "luthier_pick",
    "gibson_thumb",
]

STAGE_SUFFIXES = [
    "model",           # 合成データ事前学習
    "guitarset_ft",    # Step 2: GuitarSet FT
    "multitask",       # Step 3: + GAPS
    "multitask_3ds",   # Step 6: + GAPS + AG-PT
    "multitask_3ds_ga", # Step 9: + GAPS + Synth V2
    "multitask_4ds",   # Step 11: + GAPS + Synth V2 + IDMT
]


def discover_models():
    """利用可能な全モデルを発見"""
    models = []
    for domain in DOMAIN_NAMES:
        for suffix in STAGE_SUFFIXES:
            name = f"finetuned_{domain}_{suffix}"
            path = os.path.join(TRAINING_OUTPUT, name, "best_model.pth")
            if os.path.exists(path):
                models.append({"name": name, "path": path, "domain": domain, "stage": suffix})
    return models


def load_and_infer(model_info, features, device):
    """1モデルでの推論"""
    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=1280, rnn_type="GRU",
        rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
    )
    state_dict = torch.load(model_info["path"], map_location=device, weights_only=False)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        onset_logits, fret_logits = model(features)
        onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
        fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()

    del model, state_dict
    torch.cuda.empty_cache()

    return onset_probs, np.argmax(fret_probs, axis=-1)


def vote_and_decode(all_onset_probs, all_fret_preds, vote_threshold, onset_threshold=ONSET_THRESHOLD, vote_prob_threshold=VOTE_PROB_THRESHOLD):
    """投票→デコード"""
    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)

    binary_votes = all_onset_probs > vote_prob_threshold
    vote_counts = np.sum(binary_votes, axis=0)

    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < vote_threshold] = 0.0

    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)

    notes = _frames_to_notes(
        consensus_onset_probs, consensus_frets, tuning_pitches=None, onset_threshold=onset_threshold
    )

    for n in notes:
        n["source"] = "mega_moe"
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])

    return notes


def load_gt_notes(jams_path):
    """JAMSから正解ノートを読み込む"""
    jam = jams.load(jams_path)
    gt_notes = []
    note_midi_idx = 0
    for ann in jam.annotations:
        if ann.namespace != 'note_midi':
            continue
        string_num = 6 - note_midi_idx
        note_midi_idx += 1
        if string_num < 1 or string_num > 6:
            continue
        string_idx = 6 - string_num
        for obs in ann.data:
            midi_pitch = int(round(obs.value))
            gt_fret = midi_pitch - STANDARD_TUNING[string_idx]
            if gt_fret < 0 or gt_fret > 19:
                continue
            gt_notes.append({
                'pitch': midi_pitch,
                'start': float(obs.time),
                'duration': float(obs.duration),
                'string': string_num,
                'fret': gt_fret,
            })
    gt_notes.sort(key=lambda n: (n['start'], n['pitch']))
    return gt_notes


def match_notes(pred_notes, gt_notes, onset_tol=ONSET_TOLERANCE):
    """ノートマッチング"""
    gt_matched = [False] * len(gt_notes)
    tp_pitch = 0
    tp_sf = 0
    fp = 0

    for pred in pred_notes:
        p_onset = pred.get('start', 0)
        p_pitch = pred.get('pitch', 0)
        p_string = pred.get('string', 0)
        p_fret = pred.get('fret', 0)

        best_idx = None
        best_dt = float('inf')
        for i, gt in enumerate(gt_notes):
            if gt_matched[i]:
                continue
            dt = abs(p_onset - gt['start'])
            if dt <= onset_tol and p_pitch == gt['pitch'] and dt < best_dt:
                best_dt = dt
                best_idx = i

        if best_idx is not None:
            gt_matched[best_idx] = True
            tp_pitch += 1
            if p_string == gt_notes[best_idx]['string'] and p_fret == gt_notes[best_idx]['fret']:
                tp_sf += 1
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp_pitch, tp_sf, fp, fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. モデル発見
    models = discover_models()
    print(f"\n=== 発見されたモデル: {len(models)} ===")
    for m in models:
        print(f"  {m['name']}")

    # 2. テストファイル収集
    jams_files = sorted(glob.glob(os.path.join(ANN_DIR, '*.jams')))
    jams_files = [f for f in jams_files if os.path.basename(f)[:2] in TEST_PLAYERS]
    print(f"\nテストトラック: {len(jams_files)} tracks (Player {TEST_PLAYERS})")

    # 3. 全モデル × 全トラックの推論結果をキャッシュ
    # track_predictions[track_idx] = [(onset_probs, fret_preds), ...]
    track_data = []  # [(jams_path, mic_path, gt_notes)]
    track_predictions = []  # [[(onset_probs, fret_preds), ...], ...]

    for jams_path in jams_files:
        basename = os.path.basename(jams_path).replace('.jams', '')
        mic_path = os.path.join(MIC_DIR, basename + '_mic.wav')
        if not os.path.exists(mic_path):
            continue
        gt_notes = load_gt_notes(jams_path)
        track_data.append((jams_path, mic_path, gt_notes, basename))

    print(f"\n=== 推論開始: {len(track_data)} tracks × {len(models)} models ===")
    t0 = time.time()

    for t_idx, (jams_path, mic_path, gt_notes, basename) in enumerate(track_data):
        print(f"\n--- [{t_idx+1}/{len(track_data)}] {basename} ---")

        # CQT特徴量抽出（1回だけ）
        y, sr = librosa.load(mic_path, sr=config.SAMPLE_RATE, mono=True)
        cqt_spec = librosa.cqt(
            y=y, sr=sr, hop_length=config.HOP_LENGTH,
            fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
        )
        log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # 全モデルで推論
        preds = []
        for m_idx, model_info in enumerate(models):
            onset_probs, fret_preds = load_and_infer(model_info, features, device)
            preds.append((onset_probs, fret_preds))

        track_predictions.append(preds)
        print(f"  {len(preds)} models completed")

    elapsed = time.time() - t0
    print(f"\n=== 推論完了: {elapsed/60:.1f} min ===")

    # 4. Vote threshold スイープ
    print(f"\n{'='*80}")
    print(f"  VOTE THRESHOLD SWEEP: {len(models)} models, {len(track_data)} tracks")
    print(f"{'='*80}")

    results = []
    for vote_threshold in range(12, 31):
        total_tp_pitch = 0
        total_tp_sf = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0

        for t_idx, (jams_path, mic_path, gt_notes, basename) in enumerate(track_data):
            preds = track_predictions[t_idx]
            all_onset_probs = [p[0] for p in preds]
            all_fret_preds = [p[1] for p in preds]

            # 投票→デコード
            notes = vote_and_decode(all_onset_probs, all_fret_preds, vote_threshold)

            # CNN弦割り当て
            assigned = assign_strings_dp(
                copy.deepcopy(notes),
                tuning=STANDARD_TUNING,
                audio_path=mic_path
            )

            # マッチング
            tp_pitch, tp_sf, fp, fn = match_notes(assigned, gt_notes)
            total_tp_pitch += tp_pitch
            total_tp_sf += tp_sf
            total_fp += fp
            total_fn += fn
            total_gt += len(gt_notes)

        p_all = total_tp_pitch / max(total_tp_pitch + total_fp, 1)
        r_all = total_tp_pitch / max(total_tp_pitch + total_fn, 1)
        f1_all = 2 * p_all * r_all / max(p_all + r_all, 1e-8)
        sf_rate = total_tp_sf / max(total_tp_pitch, 1)
        e2e = total_tp_sf / max(total_gt, 1)

        results.append({
            'vote': vote_threshold,
            'f1': f1_all,
            'precision': p_all,
            'recall': r_all,
            'sf_rate': sf_rate,
            'e2e': e2e,
            'tp': total_tp_pitch,
            'fp': total_fp,
            'fn': total_fn,
        })

        marker = " ★" if f1_all > 0.8916 else ""
        print(f"  vote={vote_threshold:2d} | F1={f1_all:.4f} | P={p_all:.4f} | R={r_all:.4f} | SF={sf_rate:.4f} | E2E={e2e:.4f}{marker}")

    # 5. 最良結果
    best = max(results, key=lambda r: r['f1'])
    print(f"\n{'='*80}")
    print(f"  BEST: vote={best['vote']} | F1={best['f1']:.4f} | P={best['precision']:.4f} | R={best['recall']:.4f}")
    print(f"  比較: 35-model MoE F1=0.8916 / 7-model F1=0.8877")
    print(f"{'='*80}")

    # 結果保存
    out_path = os.path.join(r'D:\Music\nextchord-solotab\backend\benchmark', '42model_moe_sweep_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
