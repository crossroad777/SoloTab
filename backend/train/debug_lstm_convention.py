"""LSTM詳細デバッグ: 各ノートのGT/LSTM/CNN比較"""
import sys, os, glob
sys.path.insert(0, "backend")
import jams, torch
from string_assigner import _load_string_classifier, _predict_string_probs, STANDARD_TUNING, MAX_FRET
from fingering_model import load_fingering_model, _extract_features
import torch.nn.functional as F

ANNOTATION_DIR = r"D:\Music\Datasets\GuitarSet\annotation"
AUDIO_DIR = r"D:\Music\Datasets\GuitarSet\audio_mono-mic"

fm = load_fingering_model()
clf = _load_string_classifier()

jf = sorted([f for f in glob.glob(os.path.join(ANNOTATION_DIR, "*.jams"))
             if "_solo" in os.path.basename(f)])[0]
basename = os.path.splitext(os.path.basename(jf))[0]
audio = os.path.join(AUDIO_DIR, basename + "_mic.wav")

jam = jams.load(jf)
notes = []
idx = 0
for ann in jam.annotations:
    if ann.namespace != "note_midi":
        continue
    sn = 6 - idx
    idx += 1
    if sn < 1 or sn > 6:
        continue
    si = 6 - sn
    for obs in ann.data:
        p = int(round(obs.value))
        st = float(obs.time)
        dur = float(obs.duration)
        fret = p - STANDARD_TUNING[si]
        if 0 <= fret <= MAX_FRET:
            note = {"pitch": p, "start": st, "duration": dur, "gt_string": sn}
            probs = _predict_string_probs(audio, st, p)
            if probs:
                note["cnn_string_probs"] = probs
            notes.append(note)
notes.sort(key=lambda n: n["start"])

# LSTM prediction
model = fm["model"]
device = fm["device"]
feats = _extract_features(notes).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(feats)
    probs_all = F.softmax(logits[0], dim=-1)

correct = 0
total = min(20, len(notes))
print(f"File: {basename}, Notes: {len(notes)}")
print(f"{'#':>3} {'Pitch':>5} {'GT':>4} {'Pred':>5} {'Raw':>4} {'CNN':>5} {'Ok':>3}")
print("-" * 40)

for i in range(total):
    n = notes[i]
    gt = n["gt_string"]
    lstm_probs = probs_all[i].cpu().numpy()
    lstm_raw = lstm_probs.argmax() + 1  # 0-indexed -> 1-indexed

    # Physical filter
    possible = []
    for si2, op in enumerate(STANDARD_TUNING):
        s2 = 6 - si2
        fret2 = n["pitch"] - op
        if 0 <= fret2 <= 19:
            possible.append((s2, fret2, lstm_probs[s2 - 1]))
    best_s = max(possible, key=lambda x: x[2])[0] if possible else 0

    cnn = n.get("cnn_string_probs", {})
    cnn_top = max(cnn, key=cnn.get) if cnn else "-"

    ok = "OK" if best_s == gt else "X"
    if best_s == gt:
        correct += 1

    print(f"{i:3d} {n['pitch']:5d}  S{gt}   S{best_s}   S{lstm_raw}   S{cnn_top}  {ok}")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.0f}%")
