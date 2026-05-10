"""
generate_dataset.py — 大規模合成教師データ生成パイプライン
==========================================================
FluidSynth + FluidR3_GM.sf2 で多様なギター演奏パターンを生成し、
学習パイプライン互換の features.pt / labels.pt 形式で出力する。

生成カテゴリ:
  1. スケール走行（全キー×全ポジション）
  2. アルペジオ（各種パターン）
  3. コード＋フィンガーピッキング
  4. 単音メロディ（ランダム）
  5. 混合パターン（ベース＋メロディ）

Usage:
  python backend/synth/generate_dataset.py --num 5000
  python backend/synth/generate_dataset.py --num 100 --dry-run
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import librosa
import soundfile as sf

# FluidSynth DLL path
os.environ['PATH'] = r'C:\tools\fluidsynth\bin;' + os.environ.get('PATH', '')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config
from data_processing.dataset import create_frame_level_labels

# ============================================================
# Constants
# ============================================================
SF2_PATH = r"D:\Music\datasets\acoustic_guitar\FluidR3_GM.sf2"
OUTPUT_DIR = r"D:\Music\datasets\synth_v2"
SR = config.SAMPLE_RATE  # 22050

from solotab_utils import STANDARD_TUNING
MAX_FRET = config.MAX_FRETS  # 20

# GM Programs
GM_STEEL = 25
GM_NYLON = 24

# Scale patterns (intervals from root)
SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11, 12],
    "minor":       [0, 2, 3, 5, 7, 8, 10, 12],
    "pentatonic":  [0, 2, 4, 7, 9, 12],
    "blues":       [0, 3, 5, 6, 7, 10, 12],
    "dorian":      [0, 2, 3, 5, 7, 9, 10, 12],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10, 12],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11, 12],
}

CHORD_DB = {
    "C":  [(-1,-1),(3,45),(2,50),(0,55),(1,60),(0,64)],
    "D":  [(-1,-1),(-1,-1),(0,50),(2,57),(3,62),(2,66)],
    "E":  [(0,40),(2,47),(2,52),(1,56),(0,59),(0,64)],
    "F":  [(1,41),(3,48),(3,53),(2,57),(1,60),(1,65)],
    "G":  [(3,43),(2,47),(0,50),(0,55),(0,59),(3,67)],
    "A":  [(-1,-1),(0,45),(2,52),(2,57),(2,61),(0,64)],
    "Am": [(-1,-1),(0,45),(2,52),(2,57),(1,60),(0,64)],
    "Dm": [(-1,-1),(-1,-1),(0,50),(2,57),(3,62),(1,65)],
    "Em": [(0,40),(2,47),(2,52),(0,55),(0,59),(0,64)],
    "B7": [(-1,-1),(2,47),(1,51),(2,57),(0,59),(2,66)],
    "C7": [(-1,-1),(3,48),(2,52),(3,58),(1,60),(0,64)],
    "G7": [(3,43),(2,47),(0,50),(0,55),(0,59),(1,65)],
    "Am7":[(-1,-1),(0,45),(2,52),(0,55),(1,60),(0,64)],
    "Dm7":[(-1,-1),(-1,-1),(0,50),(2,57),(1,61),(1,65)],
    "Em7":[(0,40),(2,47),(0,50),(0,55),(0,59),(0,64)],
}

PROGRESSIONS = [
    ["C","G","Am","F"], ["G","Em","C","D"], ["Am","F","C","G"],
    ["C","Am","Dm","G7"], ["Em","Am","D","G"], ["Am","Dm","E","Am"],
    ["D","A","G","A"], ["C","Em","Am","Em"], ["G","C","D","G"],
    ["Am","G","F","E"], ["C","C7","F","G"], ["Em","Am7","Dm7","G7"],
]

PICKING_PATTERNS = [
    # (subdivision_index, [string_indices 0-5])  0=low E, 5=high E
    {"name": "travis",   "subs": 8, "pat": [(0,[0]),(1,[3]),(2,[1]),(3,[2]),(4,[0]),(5,[3]),(6,[1]),(7,[4,5])]},
    {"name": "arpeggio", "subs": 8, "pat": [(0,[0]),(1,[1]),(2,[2]),(3,[3]),(4,[4]),(5,[5]),(6,[4]),(7,[3])]},
    {"name": "pima",     "subs":16, "pat": [(0,[0]),(1,[3]),(2,[4]),(3,[5]),(4,[1]),(5,[3]),(6,[4]),(7,[5]),
                                            (8,[2]),(9,[3]),(10,[4]),(11,[5]),(12,[1]),(13,[3]),(14,[4]),(15,[5])]},
    {"name": "ballad",   "subs": 8, "pat": [(0,[0,3]),(1,[4]),(2,[5]),(3,[4]),(4,[1,3]),(5,[4]),(6,[5]),(7,[4])]},
    {"name": "waltz",    "subs": 6, "pat": [(0,[0]),(1,[3,4,5]),(2,[3,4,5]),(3,[1]),(4,[3,4,5]),(5,[3,4,5])]},
]


# ============================================================
# FluidSynth Renderer
# ============================================================
class Renderer:
    def __init__(self, sf2_path=SF2_PATH, sr=SR, program=GM_STEEL):
        import fluidsynth
        self.sr = sr
        self.sf2_path = sf2_path
        self.program = program

    def render(self, note_events):
        """note_events: list of {pitch, start, duration, velocity, string, fret}"""
        import fluidsynth
        if not note_events:
            return np.zeros(self.sr, dtype=np.float32)

        events = sorted(note_events, key=lambda e: e["start"])
        max_end = max(e["start"] + e["duration"] for e in events)
        total_ms = int((max_end + 1.5) * 1000)

        fs = fluidsynth.Synth(samplerate=float(self.sr))
        sfid = fs.sfload(self.sf2_path)
        fs.program_select(0, sfid, 0, self.program)
        fs.setting('synth.reverb.active', 1)
        fs.setting('synth.reverb.room-size', np.random.uniform(0.2, 0.6))
        fs.setting('synth.reverb.level', np.random.uniform(0.1, 0.4))

        active = []
        chunks = []
        cur_ms = 0

        for ev in events:
            t_ms = int(ev["start"] * 1000)
            pitch = ev["pitch"]
            vel = int(np.clip(ev.get("velocity", 0.7) * 127, 20, 127))
            dur_ms = int(ev["duration"] * 1000)

            if t_ms > cur_ms:
                n = int(self.sr * (t_ms - cur_ms) / 1000)
                if n > 0:
                    chunks.append(np.array(fs.get_samples(n), dtype=np.float32) / 32768.0)
                cur_ms = t_ms

            still = []
            for end_t, p in active:
                if end_t <= cur_ms:
                    fs.noteoff(0, p)
                else:
                    still.append((end_t, p))
            active = still

            fs.noteon(0, pitch, vel)
            active.append((t_ms + dur_ms, pitch))

        rem = total_ms - cur_ms
        if rem > 0:
            n = int(self.sr * rem / 1000)
            if n > 0:
                chunks.append(np.array(fs.get_samples(n), dtype=np.float32) / 32768.0)

        fs.delete()
        if not chunks:
            return np.zeros(self.sr, dtype=np.float32)

        audio = np.concatenate(chunks)
        mono = audio.reshape(-1, 2).mean(axis=1)
        peak = np.max(np.abs(mono))
        if peak > 0.01:
            mono = mono / peak * np.random.uniform(0.7, 0.95)
        return mono.astype(np.float32)


# ============================================================
# Note Generators
# ============================================================
def pitch_to_string_fret(midi_pitch):
    """Returns (string_idx, fret) or None. string_idx: 0=lowE..5=highE"""
    candidates = []
    for s, open_p in enumerate(STANDARD_TUNING):
        fret = midi_pitch - open_p
        if 0 <= fret <= MAX_FRET:
            candidates.append((s, fret))
    return min(candidates, key=lambda x: x[1]) if candidates else None


def gen_scale(rng):
    """Generate a scale run."""
    root = rng.integers(40, 65)
    scale_name = rng.choice(list(SCALES.keys()))
    intervals = SCALES[scale_name]
    bpm = rng.uniform(60, 160)
    note_dur_beats = rng.choice([0.25, 0.5, 1.0])
    beat_dur = 60.0 / bpm
    note_dur = beat_dur * note_dur_beats

    # Go up then down
    pitches = [root + i for i in intervals]
    down = list(reversed(pitches[:-1]))
    full = pitches + down

    # Optionally repeat
    reps = rng.integers(1, 4)
    sequence = full * reps

    events = []
    t = rng.uniform(0.0, 0.3)
    for p in sequence:
        sf = pitch_to_string_fret(p)
        if sf is None:
            t += note_dur
            continue
        s, f = sf
        events.append({
            "pitch": int(p), "start": float(t), "duration": float(note_dur * rng.uniform(0.7, 1.0)),
            "velocity": float(rng.uniform(0.5, 0.9)), "string": int(s), "fret": int(f),
        })
        t += note_dur + rng.uniform(-0.02, 0.02) * note_dur
    return events


def gen_arpeggio(rng):
    """Generate chord arpeggios."""
    prog_idx = rng.integers(len(PROGRESSIONS))
    prog = PROGRESSIONS[prog_idx]
    bpm = rng.uniform(70, 140)
    beat_dur = 60.0 / bpm
    pattern = PICKING_PATTERNS[rng.integers(len(PICKING_PATTERNS))]
    subs = pattern["subs"]
    pat = pattern["pat"]
    measures_per_chord = rng.integers(1, 4)
    time_sig_beats = 3 if pattern["name"] == "waltz" else 4
    measure_dur = beat_dur * time_sig_beats
    sub_dur = measure_dur / subs

    events = []
    t = rng.uniform(0.0, 0.2)

    for chord_name in prog:
        chord = CHORD_DB.get(chord_name)
        if chord is None:
            t += measure_dur * measures_per_chord
            continue

        for _ in range(measures_per_chord):
            for sub_idx, strings in pat:
                nt = t + sub_idx * sub_dur
                nd = sub_dur * rng.uniform(1.2, 2.0)
                for s in strings:
                    fret, pitch = chord[s]
                    if fret < 0:
                        continue
                    events.append({
                        "pitch": int(pitch), "start": float(nt + rng.uniform(-0.01, 0.01)),
                        "duration": float(nd), "velocity": float(rng.uniform(0.45, 0.85)),
                        "string": int(s), "fret": int(fret),
                    })
            t += measure_dur
    return events


def gen_melody(rng):
    """Generate random single-note melody."""
    n_notes = rng.integers(10, 50)
    bpm = rng.uniform(60, 150)
    beat_dur = 60.0 / bpm
    root = rng.integers(40, 72)
    scale_name = rng.choice(list(SCALES.keys()))
    intervals = SCALES[scale_name]

    # Build available pitches across 2 octaves
    available = []
    for octave in range(3):
        for i in intervals:
            p = root + i + 12 * octave
            sf = pitch_to_string_fret(p)
            if sf:
                available.append((p, sf[0], sf[1]))
    if not available:
        return []

    events = []
    t = rng.uniform(0.0, 0.3)
    for _ in range(n_notes):
        idx = rng.integers(len(available))
        p, s, f = available[idx]
        dur = beat_dur * rng.choice([0.25, 0.5, 0.5, 1.0, 1.0, 2.0])
        events.append({
            "pitch": int(p), "start": float(t), "duration": float(dur * rng.uniform(0.6, 1.0)),
            "velocity": float(rng.uniform(0.4, 0.95)), "string": int(s), "fret": int(f),
        })
        t += dur + rng.uniform(-0.02, 0.03) * dur
        # Occasional rest
        if rng.random() < 0.15:
            t += beat_dur * rng.choice([0.5, 1.0])
    return events


def gen_chord_strum(rng):
    """Generate strummed chords."""
    prog_idx = rng.integers(len(PROGRESSIONS))
    prog = PROGRESSIONS[prog_idx]
    bpm = rng.uniform(70, 140)
    beat_dur = 60.0 / bpm
    strum_spread = rng.uniform(0.01, 0.04)

    events = []
    t = rng.uniform(0.0, 0.2)

    for chord_name in prog:
        chord = CHORD_DB.get(chord_name)
        if chord is None:
            t += beat_dur * 4
            continue

        n_strums = rng.integers(2, 6)
        for _ in range(n_strums):
            dur = beat_dur * rng.choice([1.0, 2.0, 4.0])
            vel_base = rng.uniform(0.5, 0.9)
            strum_t = 0
            for s in range(6):
                fret, pitch = chord[s]
                if fret < 0:
                    continue
                events.append({
                    "pitch": int(pitch), "start": float(t + strum_t),
                    "duration": float(dur * rng.uniform(0.8, 1.0)),
                    "velocity": float(vel_base + rng.uniform(-0.05, 0.05)),
                    "string": int(s), "fret": int(fret),
                })
                strum_t += strum_spread * rng.uniform(0.5, 1.5)
            t += dur
    return events


def gen_mixed(rng):
    """Generate bass + melody (fingerstyle)."""
    prog_idx = rng.integers(len(PROGRESSIONS))
    prog = PROGRESSIONS[prog_idx]
    bpm = rng.uniform(70, 120)
    beat_dur = 60.0 / bpm
    scale_name = rng.choice(["major", "minor", "pentatonic"])

    events = []
    t = rng.uniform(0.0, 0.2)

    for chord_name in prog:
        chord = CHORD_DB.get(chord_name)
        if chord is None:
            t += beat_dur * 4
            continue

        # Bass notes (strings 0-1)
        for beat in range(4):
            bass_s = rng.integers(0, 2)
            fret, pitch = chord[bass_s]
            if fret >= 0:
                events.append({
                    "pitch": int(pitch), "start": float(t + beat * beat_dur),
                    "duration": float(beat_dur * 0.9),
                    "velocity": float(rng.uniform(0.6, 0.85)),
                    "string": int(bass_s), "fret": int(fret),
                })

        # Melody on top strings (3-5)
        root_pitch = None
        for s in range(6):
            f, p = chord[s]
            if f >= 0:
                root_pitch = p
                break
        if root_pitch is None:
            t += beat_dur * 4
            continue

        intervals = SCALES[scale_name]
        mel_pitches = []
        for i in intervals:
            p = root_pitch + i
            sf_r = pitch_to_string_fret(p)
            if sf_r and sf_r[0] >= 3:
                mel_pitches.append((p, sf_r[0], sf_r[1]))

        n_mel = rng.integers(3, 9)
        for i in range(n_mel):
            if not mel_pitches:
                break
            idx = rng.integers(len(mel_pitches))
            p, s, f = mel_pitches[idx]
            mel_t = t + rng.uniform(0, 4) * beat_dur
            events.append({
                "pitch": int(p), "start": float(mel_t),
                "duration": float(beat_dur * rng.uniform(0.3, 1.5)),
                "velocity": float(rng.uniform(0.5, 0.9)),
                "string": int(s), "fret": int(f),
            })
        t += beat_dur * 4
    return events


GENERATORS = [
    ("scale", gen_scale, 0.20),
    ("arpeggio", gen_arpeggio, 0.25),
    ("melody", gen_melody, 0.20),
    ("chord_strum", gen_chord_strum, 0.15),
    ("mixed", gen_mixed, 0.20),
]


# ============================================================
# WAV → features.pt / labels.pt
# ============================================================
def wav_to_features(audio, sr=SR):
    """Compute CQT features matching the training pipeline."""
    cqt = librosa.cqt(
        y=audio, sr=sr,
        hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT,
        n_bins=config.N_BINS_CQT,
        bins_per_octave=config.BINS_PER_OCTAVE_CQT,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return torch.tensor(log_cqt, dtype=torch.float32)


def events_to_raw_labels(events):
    """Convert note events to raw_labels tensor [N, 5]: onset, offset, string, fret, pitch"""
    if not events:
        return torch.zeros((0, 5), dtype=torch.float32)
    rows = []
    for e in events:
        rows.append([
            e["start"],
            e["start"] + e["duration"],
            float(e["string"]),
            float(e["fret"]),
            float(e["pitch"]),
        ])
    arr = np.array(rows, dtype=np.float32)
    arr = arr[arr[:, 0].argsort()]
    return torch.from_numpy(arr)


# ============================================================
# Main Pipeline
# ============================================================
def generate_dataset(num_samples, output_dir, dry_run=False, sf2_path=None, program=None):
    os.makedirs(output_dir, exist_ok=True)

    renderer = Renderer(sf2_path=sf2_path or SF2_PATH, sr=SR, program=program if program is not None else GM_STEEL)
    rng = np.random.default_rng(42)

    # Compute category counts
    gen_names = [g[0] for g in GENERATORS]
    gen_funcs = [g[1] for g in GENERATORS]
    gen_weights = np.array([g[2] for g in GENERATORS])
    gen_weights /= gen_weights.sum()
    gen_counts = (gen_weights * num_samples).astype(int)
    gen_counts[-1] = num_samples - gen_counts[:-1].sum()

    print(f"\n{'='*60}")
    print(f"  Synthetic Dataset Generation")
    print(f"{'='*60}")
    print(f"  Output:    {output_dir}")
    print(f"  Samples:   {num_samples}")
    print(f"  SF2:       {renderer.sf2_path}")
    print(f"  SR:        {SR}")
    print(f"  Dry run:   {dry_run}")
    for name, count in zip(gen_names, gen_counts):
        print(f"    {name}: {count}")
    print()

    # Build generation schedule
    schedule = []
    for i, (name, func, _) in enumerate(GENERATORS):
        for _ in range(gen_counts[i]):
            schedule.append((name, func))
    rng.shuffle(schedule)

    t0 = time.time()
    total_dur = 0
    total_notes = 0
    stats = {n: 0 for n in gen_names}
    ids = []
    errors = 0

    for idx, (cat_name, gen_func) in enumerate(schedule):
        try:
            events = gen_func(rng)
            if not events or len(events) < 3:
                events = gen_melody(rng)
            if not events:
                errors += 1
                continue

            # Ensure start >= 0
            for e in events:
                e["start"] = max(0, e["start"])

            sample_id = f"synth_v2_{idx:05d}"

            if not dry_run:
                # Randomly choose steel or nylon
                prog = GM_STEEL if rng.random() > 0.3 else GM_NYLON
                renderer.program = prog

                audio = renderer.render(events)
                if len(audio) < SR:
                    errors += 1
                    continue

                # Compute features
                features = wav_to_features(audio)
                raw_labels = events_to_raw_labels(events)

                # Save
                torch.save(features, os.path.join(output_dir, f"{sample_id}_features.pt"))
                torch.save(raw_labels, os.path.join(output_dir, f"{sample_id}_labels.pt"))

                # Also save WAV + JSON for inspection
                if idx < 20:
                    inspect_dir = os.path.join(output_dir, "_inspect")
                    os.makedirs(inspect_dir, exist_ok=True)
                    sf.write(os.path.join(inspect_dir, f"{sample_id}.wav"), audio, SR)
                    with open(os.path.join(inspect_dir, f"{sample_id}.json"), "w") as f:
                        json.dump({"category": cat_name, "notes": [
                            {"pitch": e["pitch"], "start": round(e["start"], 4),
                             "end": round(e["start"]+e["duration"], 4),
                             "string": e["string"], "fret": e["fret"]}
                            for e in events
                        ]}, f, indent=2)

                total_dur += len(audio) / SR
            else:
                total_dur += sum(e["start"] + e["duration"] for e in events[-1:])

            total_notes += len(events)
            stats[cat_name] += 1
            ids.append(sample_id)

            if (idx + 1) % 100 == 0 or idx == 0:
                elapsed = time.time() - t0
                speed = total_dur / elapsed if elapsed > 0 else 0
                print(f"  [{idx+1}/{num_samples}] {cat_name:12s} | "
                      f"{len(events):3d} notes | "
                      f"total: {total_dur/60:.1f}min | "
                      f"{speed:.1f}x RT | errors: {errors}")

        except Exception as ex:
            print(f"  [ERROR] {idx}: {ex}")
            errors += 1

    # Save ID list
    ids_path = os.path.join(output_dir, "train_ids.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(ids))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Generation Complete!")
    print(f"{'='*60}")
    print(f"  Samples:  {len(ids)} (errors: {errors})")
    print(f"  Duration: {total_dur/60:.1f} min ({total_dur/3600:.2f} h)")
    print(f"  Notes:    {total_notes}")
    print(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if elapsed > 0:
        print(f"  Speed:    {total_dur/elapsed:.1f}x realtime")
    print(f"  IDs file: {ids_path}")
    print(f"  Categories: {stats}")
    return ids_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic guitar training data")
    parser.add_argument("--num", type=int, default=5000, help="Number of samples")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Generate events only (no audio)")
    parser.add_argument("--sf2", type=str, default=None, help="SoundFont path (override default)")
    parser.add_argument("--program", type=int, default=None, help="MIDI program number (default: 25=Steel Guitar)")
    args = parser.parse_args()

    generate_dataset(args.num, args.output, args.dry_run, sf2_path=args.sf2, program=args.program)
