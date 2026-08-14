import os, sys, glob, json, time
import numpy as np
import torch
import librosa
from scipy import stats

sys.path.insert(0, r'D:\Music\chordlink-solotab\backend')
try:
    from string_assigner import assign_strings_dp
    from finger_assigner import assign_fingers
except ImportError as e:
    print(f"Error importing Viterbi modules: {e}")
    sys.exit(1)

try:
    import mir_eval
except ImportError:
    print("mir_eval required: pip install mir_eval")
    sys.exit(1)

mt_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "music-transcription", "python")
sys.path.insert(0, mt_python_dir)
import config
from model import architecture
from guitar_transcriber import _frames_to_notes

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")

VOTE_THRESHOLD = 5
ONSET_THRESHOLD = 0.8
VOTE_PROB_THRESHOLD = 0.5

MODELS = [
    "finetuned_martin_finger_guitarset_ft", "finetuned_taylor_finger_guitarset_ft",
    "finetuned_luthier_finger_guitarset_ft", "finetuned_martin_pick_guitarset_ft",
    "finetuned_taylor_pick_guitarset_ft", "finetuned_luthier_pick_guitarset_ft",
]

TARGET_TRACKS = [
    "05_Jazz2-187-F#_comp",
    "00_SS1-68-E_comp",
    "03_BN1-147-Gb_comp",
    "01_Funk1-97-C_comp",
    "05_Jazz2-187-F#_solo",
    "00_Rock2-142-D_solo",
    "04_Rock1-130-A_solo",
    "05_SS2-88-F_solo",
    "02_Funk2-119-G_comp"
]

def load_jams_notes(jams_path):
    with open(jams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            for d in ann.get('data', []):
                start = float(d.get('time', 0.0))
                dur = float(d.get('duration', 0.0))
                pitch = int(round(float(d.get('value', 0.0))))
                notes.append({"start": start, "end": start + dur, "pitch": pitch})
    notes.sort(key=lambda x: x['start'])
    return notes

def to_mireval(notes):
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = np.array([[n['start'], n['end']] for n in notes], dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((n['pitch'] - 69.0) / 12.0)) for n in notes], dtype=float)
    return intervals, pitches_hz

def run_inference(wav_path, device):
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(y=y, sr=sr, hop_length=config.HOP_LENGTH,
                           fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
                           bins_per_octave=config.BINS_PER_OCTAVE_CQT)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    all_onset_probs = []
    all_fret_preds = []
    for model_dir in MODELS:
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            continue
        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=1280, rnn_type="GRU",
            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
        )
        sd = torch.load(model_path, map_location=device, weights_only=False)
        if list(sd.keys())[0].startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.to(device).eval()
        with torch.no_grad():
            model_output = model(features)
            onset_probs = torch.sigmoid(model_output[0][0]).cpu().numpy()
            fret_probs = torch.softmax(model_output[1][0], dim=-1).cpu().numpy()
        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        del model, sd
        torch.cuda.empty_cache()

    return np.array(all_onset_probs), np.array(all_fret_preds)

def decode(all_onset_probs, all_fret_preds):
    binary_votes = all_onset_probs > VOTE_PROB_THRESHOLD
    vote_counts = np.sum(binary_votes, axis=0)
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < VOTE_THRESHOLD] = 0.0
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)
    notes = _frames_to_notes(consensus_onset_probs, consensus_frets, tuning_pitches=None, onset_threshold=ONSET_THRESHOLD)
    for n in notes:
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])
    return notes

def run_dadgad_test():
    print("\n--- Running Special Case: DADGAD ---")
    tuning = [38, 45, 50, 55, 57, 62]  # D2, A2, D3, G3, A3, D4
    
    # Test case 3: DADGAD Open Arpeggio (All open strings, let ring)
    print("  [Subtest] DADGAD Open Arpeggio")
    notes_open = [
        {"start": 0.0, "end": 3.0, "pitch": 38}, 
        {"start": 0.5, "end": 3.0, "pitch": 45}, 
        {"start": 1.0, "end": 3.0, "pitch": 50},
        {"start": 1.5, "end": 3.0, "pitch": 55}, 
        {"start": 2.0, "end": 3.0, "pitch": 57}, 
        {"start": 2.5, "end": 3.0, "pitch": 62}
    ]
    try:
        assigned_open = assign_strings_dp(notes_open, tuning=tuning)
        assigned_open = assign_fingers(assigned_open)
        open_ok = all(n.get('fret') == 0 for n in assigned_open)
        if not open_ok:
            print("    [FAIL] Not all notes were assigned to open strings (fret=0).")
            print("    Output:", [(n['pitch'], n['string'], n['fret']) for n in assigned_open])
            return False
    except Exception as e:
        print(f"    [FAIL] Crashed: {e}")
        return False
        
    # Test case 3b: DADGAD Drone (fingers 1 and 2)
    print("  [Subtest] DADGAD Drone (fingers 1 and 2)")
    notes_drone = [
        {"start": 0.0, "end": 0.5, "pitch": 38}, {"start": 0.0, "end": 0.5, "pitch": 57}, {"start": 0.0, "end": 0.5, "pitch": 62},
        {"start": 0.5, "end": 1.0, "pitch": 64},
        {"start": 1.0, "end": 1.5, "pitch": 38}, {"start": 1.0, "end": 1.5, "pitch": 62}, {"start": 1.0, "end": 1.5, "pitch": 57},
        {"start": 1.5, "end": 2.0, "pitch": 60}
    ]
    try:
        assigned_drone = assign_strings_dp(notes_drone, tuning=tuning)
        assigned_drone = assign_fingers(assigned_drone)
        fingers = [n.get('finger') for n in assigned_drone if n.get('finger', 0) > 0]
        fingers_ok = all(f in [1, 2] for f in fingers)
        if not fingers_ok:
            print(f"    [FAIL] Fingers used were not (1, 2): {fingers}")
            return False
    except Exception as e:
        print(f"    [FAIL] Crashed: {e}")
        return False

    print("  [SUCCESS] DADGAD tests passed.")
    return True

def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    jams_files = []
    for track in TARGET_TRACKS:
        p = os.path.join(ANNOTATIONS_DIR, f"{track}.jams")
        if os.path.exists(p):
            jams_files.append(p)
        else:
            print(f"Warning: Track {track} not found in {ANNOTATIONS_DIR}")
            
    if not jams_files:
        print("No target tracks found. Aborting.")
        sys.exit(1)

    print(f"Total target tracks: {len(jams_files)}")
    results = {}
    viterbi_success = True

    for i, jams_path in enumerate(jams_files):
        base = os.path.basename(jams_path).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        print(f"\n[{i+1}/{len(jams_files)}] {base}")
        
        gt_notes = load_jams_notes(jams_path)
        if not gt_notes:
            print("-> No GT, skip")
            continue

        try:
            # 1. CRNN Inference & F1 Score
            t0 = time.time()
            all_onset_probs, all_fret_preds = run_inference(wav_path, device)
            pred_notes = decode(all_onset_probs, all_fret_preds)
            t1 = time.time()
            
            ref_intervals, ref_pitches = to_mireval(gt_notes)
            if not pred_notes:
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                est_intervals, est_pitches = to_mireval(pred_notes)
                p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals, ref_pitches, est_intervals, est_pitches,
                    onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
                )
            
            # 2. Viterbi Pipeline Execution (Freeze & Crash check)
            t2 = time.time()
            for n in pred_notes:
                if 'end' not in n: n['end'] = n['start'] + 0.1
            assigned = assign_strings_dp(pred_notes, max_fret=24)
            assigned = assign_fingers(assigned)
            t3 = time.time()

            results[base] = f1
            print(f"  F1={f1:.4f} P={p:.4f} R={r:.4f} (CRNN: {t1-t0:.1f}s | Viterbi: {t3-t2:.1f}s)")
        
        except Exception as e:
            print(f"  [ERROR] Processing {base} failed: {e}")
            viterbi_success = False

    # 3. DADGAD Test
    dadgad_ok = run_dadgad_test()
    if not dadgad_ok:
        viterbi_success = False

    # 4. Final Evaluation / Baseline Generation
    print("\n" + "="*50)
    print(" BASELINE RESULTS FOR MINI-BENCHMARK")
    print("="*50)
    
    elapsed = time.time() - start_time
    all_f1 = list(results.values())
    mean_f1 = np.mean(all_f1) if all_f1 else 0.0
    
    print(f"Total Elapsed Time: {elapsed:.1f}s")
    print(f"Overall F1 Score  : {mean_f1:.4f}")
    
    print("\n[Layer 1] OVERALL THRESHOLD:")
    print(f"  Target: >= {mean_f1 - 0.02:.4f}")
    
    print("\n[Layer 2] INDIVIDUAL THRESHOLDS:")
    for track in TARGET_TRACKS:
        if track in results:
            print(f"  {track}: >= {results[track] - 0.05:.4f} (Baseline: {results[track]:.4f})")
            
    print("\n[DADGAD & Freeze Requirements]")
    print(f"  05_Jazz2-187-F#_comp completed Viterbi without freeze: {viterbi_success}")
    print(f"  DADGAD tests passed: {dadgad_ok}")

    if not viterbi_success or not dadgad_ok:
        print("\n[RESULT] BASELINE MEASUREMENT FAILED DUE TO CRASHES. \u274c")
        sys.exit(1)
        
    print("\n[RESULT] BASELINE MEASUREMENT COMPLETE. \U0001F389")
    sys.exit(0)

if __name__ == '__main__':
    main()
