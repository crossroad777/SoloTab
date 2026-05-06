"""
preprocess_agpt.py — AG-PT-set preprocessing
=============================================
AG-PT-set CSV annotations + WAV → CQT features + raw_labels tensors.
Output format is identical to GuitarSet/GAPS for seamless integration.
"""
import os, sys, io, csv, re
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ["TQDM_ASCII"] = " 123456789#"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)
import config

AGPT_DIR = r"D:\Music\datasets\AG-PT-set\aGPTset"
AGPT_AUDIO_DIR = os.path.join(AGPT_DIR, "data", "audio")
AGPT_LABELS_CSV = os.path.join(AGPT_DIR, "metadata", "note_labels.csv")
AGPT_PROCESSED_DIR = os.path.join(AGPT_DIR, "_processed")

# AG-PT-set string mapping: their string_number 1=high E → our string 5
# their string_number 6=low E → our string 0
AGPT_STRING_TO_OURS = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}

# Open string MIDI pitches (our indexing)
OPEN_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

# Estimated note duration for single-note recordings
DEFAULT_NOTE_DURATION = 0.15  # seconds


def hz_to_midi(freq_hz):
    """Convert frequency in Hz to MIDI note number."""
    if freq_hz <= 0:
        return 0
    return 69 + 12 * np.log2(freq_hz / 440.0)


def build_labels_index(csv_path):
    """
    Parse note_labels.csv and group notes by audio_file_path.
    Returns dict: {filename: [(onset_sec, string_ours, fret, pitch_midi), ...]}
    """
    file_notes = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['audio_file_path'].strip()
            onset = float(row['onset_label_seconds'])
            is_percussive = row['isPercussive'].strip() == 'True'

            if is_percussive:
                # Percussive: onset only, no pitch/string/fret
                # Use string=0, fret=silence to signal onset-only
                file_notes.setdefault(fname, []).append(
                    (onset, -1, -1, 0.0)  # sentinel values
                )
                continue

            # Pitched techniques
            pitch_hz_str = row.get('pitch_midi', 'None').strip()
            string_agpt_str = row.get('string_number', 'None').strip()

            if pitch_hz_str == 'None' or string_agpt_str == 'None':
                continue

            pitch_hz = float(pitch_hz_str)
            pitch_midi = hz_to_midi(pitch_hz)
            string_agpt = int(string_agpt_str)
            string_ours = AGPT_STRING_TO_OURS.get(string_agpt)
            if string_ours is None:
                continue

            open_pitch = OPEN_MIDI[string_ours]
            fret = int(round(pitch_midi)) - open_pitch
            if fret < 0:
                fret = 0
            if fret > config.MAX_FRETS:
                continue  # out of range

            file_notes.setdefault(fname, []).append(
                (onset, string_ours, fret, pitch_midi)
            )

    return file_notes


def notes_to_raw_labels(notes_list):
    """
    Convert list of (onset, string, fret, pitch) to raw_labels tensor.
    For percussive notes (string=-1), we still include them for onset training
    but mark string/fret specially.
    """
    pitched = []
    for onset, string_idx, fret, pitch in notes_list:
        if string_idx < 0:
            continue  # skip percussive for now (no string/fret)
        offset = onset + DEFAULT_NOTE_DURATION
        pitched.append([onset, offset, float(string_idx), float(fret), pitch])

    if not pitched:
        return torch.zeros((0, 5), dtype=torch.float32)

    arr = np.array(pitched, dtype=np.float32)
    arr = arr[arr[:, 0].argsort()]
    return torch.from_numpy(arr)


def preprocess_agpt():
    os.makedirs(AGPT_PROCESSED_DIR, exist_ok=True)

    print("Loading AG-PT-set annotations...")
    file_notes = build_labels_index(AGPT_LABELS_CSV)
    print(f"  Files with labels: {len(file_notes)}")

    # Filter: only process files that exist + have pitched notes
    audio_files = []
    for fname, notes in file_notes.items():
        wav_path = os.path.join(AGPT_AUDIO_DIR, fname)
        if os.path.exists(wav_path):
            pitched_count = sum(1 for n in notes if n[1] >= 0)
            if pitched_count > 0:
                audio_files.append((fname, wav_path, notes))

    print(f"  Files with pitched notes + audio: {len(audio_files)}")

    train_ids = []
    processed = 0
    skipped = 0
    errors = 0

    print(f"\n{'='*60}")
    print(f"  AG-PT-set Preprocessing: {len(audio_files)} tracks")
    print(f"{'='*60}")

    for fname, wav_path, notes in tqdm(audio_files, desc="Processing AG-PT", unit="track"):
        track_id = Path(fname).stem
        feat_path = os.path.join(AGPT_PROCESSED_DIR, f"{track_id}_features.pt")
        label_path = os.path.join(AGPT_PROCESSED_DIR, f"{track_id}_labels.pt")

        if os.path.exists(feat_path) and os.path.exists(label_path):
            train_ids.append(track_id)
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
            cqt = librosa.cqt(
                y=audio, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
                bins_per_octave=config.BINS_PER_OCTAVE_CQT,
            )
            log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            features = torch.tensor(log_cqt, dtype=torch.float32)

            raw_labels = notes_to_raw_labels(notes)
            if raw_labels.shape[0] == 0:
                errors += 1
                continue

            torch.save(features, feat_path)
            torch.save(raw_labels, label_path)
            train_ids.append(track_id)
            processed += 1

        except Exception as e:
            print(f"\n  Error: {track_id}: {e}")
            errors += 1

    # Save ID list
    with open(os.path.join(AGPT_PROCESSED_DIR, "train_ids.txt"), "w") as f:
        f.write("\n".join(train_ids))

    print(f"\n  Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Output: {AGPT_PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_agpt()
