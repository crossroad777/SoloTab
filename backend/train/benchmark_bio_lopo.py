"""
Biomechanical Viterbi LOPO: 真の汎化精度測定
=============================================
各foldでCNNを再学習し、未見プレイヤーに対して
生体力学Viterbiを適用。これが本当の精度。
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import jams
from collections import Counter

from string_classifier import (
    StringClassifierCNN, StringDataset, compute_cqt,
    STANDARD_TUNING, SR, HOP_LENGTH, CONTEXT_FRAMES, N_BINS,
    ANNOTATION_DIR, MIC_AUDIO_DIR
)
MAX_FRET = 19

FINGER_EASE = {0: 0.40, 1: 0.35, 2: 0.30, 3: 0.25, 4: 0.10}
MAX_SPAN = {(1,2):4, (1,3):5, (1,4):6, (2,3):3, (2,4):4, (3,4):3}

# Best params from benchmark_biomechanical_v2
W_STRING = 0.3; W_POS = 0.5; W_EASE = 0.5; W_OPEN = 1.0


def build_dataset_by_player():
    """プレイヤー別データセット + 評価用メタデータ"""
    jams_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.jams")))
    player_data = {}     # for CNN training
    player_eval = {}     # for Viterbi evaluation (per-file notes with GT)

    for jf in jams_files:
        basename = os.path.basename(jf).replace(".jams", "")
        player = basename[:2]
        mic_path = os.path.join(MIC_AUDIO_DIR, basename + "_mic.wav")
        if not os.path.exists(mic_path):
            continue
        is_solo = "_solo" in basename

        try:
            cqt = compute_cqt(mic_path)
            jam = jams.load(jf)
        except:
            continue

        idx = 0
        file_notes = []
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue
            sn = 6 - idx; idx += 1
            if sn < 1 or sn > 6: continue
            si = 6 - sn

            for obs in ann.data:
                onset = float(obs.time)
                pitch = int(round(obs.value))
                fret = pitch - STANDARD_TUNING[si]

                frame_idx = int(onset * SR / HOP_LENGTH)
                half_ctx = CONTEXT_FRAMES // 2
                if frame_idx - half_ctx < 0 or frame_idx + half_ctx >= cqt.shape[1]:
                    continue

                patch = cqt[:, frame_idx - half_ctx:frame_idx + half_ctx + 1]

                if player not in player_data:
                    player_data[player] = []
                player_data[player].append({
                    "patch": patch.astype(np.float32),
                    "pitch": pitch,
                    "string": sn,
                })

                if is_solo and 0 <= fret <= MAX_FRET:
                    file_notes.append({
                        "pitch": pitch, "start": onset,
                        "gt_string": sn, "gt_fret": fret,
                        "patch": patch.astype(np.float32),
                    })

        if is_solo and file_notes:
            file_notes.sort(key=lambda n: n["start"])
            if player not in player_eval:
                player_eval[player] = []
            player_eval[player].append({
                "basename": basename, "notes": file_notes
            })

    return player_data, player_eval


def train_cnn(train_features, epochs=50, batch_size=64):
    """CNNを学習してモデルを返す"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(StringDataset(train_features, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    model = StringClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = 1e9; best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0; total_n = 0
        for patches, pitches, labels in train_loader:
            patches, pitches, labels = patches.to(device), pitches.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(patches, pitches)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
        avg_loss = total_loss / total_n
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, device


def predict_probs_with_model(model, device, patch, pitch):
    """学習済みモデルでCNN確率を取得"""
    with torch.no_grad():
        p_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
        # pitch must be (batch, 1) normalized
        pitch_norm = torch.FloatTensor([[pitch / 127.0]]).to(device)
        out = model(p_tensor, pitch_norm)
        probs_raw = torch.softmax(out, dim=1).cpu().numpy()[0]
    return {s + 1: float(probs_raw[s]) for s in range(6)}


def get_candidates(pitch):
    cands = []
    for si, op in enumerate(STANDARD_TUNING):
        s = 6 - si; f = pitch - op
        if 0 <= f <= 19:
            if f == 0:
                cands.append((s, f, 0))
            else:
                for finger in range(1, 5):
                    cands.append((s, f, finger))
    return cands


def transition_cost(s1, f1, fig1, s2, f2, fig2, dt):
    cost = W_STRING * abs(s1 - s2)
    if fig1 == 0 or fig2 == 0:
        if fig2 == 0: cost -= W_OPEN
        return cost
    pos1 = f1 - (fig1 - 1); pos2 = f2 - (fig2 - 1)
    ps = abs(pos2 - pos1)
    if ps > 0:
        tf = 1.0 / max(dt, 0.05)
        cost += W_POS * ps * min(tf, 5.0)
    if fig1 == fig2 and f1 != f2: cost += 2.0
    if s1 == s2:
        if (fig2 > fig1 and f2 < f1) or (fig2 < fig1 and f2 > f1): cost += 10.0
    if fig1 != fig2 and fig1 > 0 and fig2 > 0:
        span = abs(f2 - f1)
        pair = (min(fig1, fig2), max(fig1, fig2))
        ms = MAX_SPAN.get(pair, 6)
        if span > ms: cost += 1.0 * (span - ms)
    return cost


def viterbi_bio(notes, cnn_probs_list):
    n = len(notes)
    if n == 0: return []
    candidates = [get_candidates(notes[i]["pitch"]) or [(1,0,0)] for i in range(n)]
    INF = 1e9; dp = []; backptr = []

    probs0 = cnn_probs_list[0]
    costs0 = []
    for (s, f, fig) in candidates[0]:
        em = -np.log(max(probs0.get(s, 1e-10), 1e-10))
        ease = -W_EASE * FINGER_EASE.get(fig, 0.1)
        costs0.append(em + ease)
    dp.append(costs0); backptr.append([-1]*len(candidates[0]))

    for i in range(1, n):
        dt = notes[i]["start"] - notes[i-1]["start"]
        probs_i = cnn_probs_list[i]
        costs_i = []; bp_i = []
        for j, (sj, fj, figj) in enumerate(candidates[i]):
            em = -np.log(max(probs_i.get(sj, 1e-10), 1e-10))
            ease = -W_EASE * FINGER_EASE.get(figj, 0.1)
            bc = INF; bp = 0
            for k, (sk, fk, figk) in enumerate(candidates[i-1]):
                c = dp[i-1][k] + transition_cost(sk, fk, figk, sj, fj, figj, dt) + em + ease
                if c < bc: bc = c; bp = k
            costs_i.append(bc); bp_i.append(bp)
        dp.append(costs_i); backptr.append(bp_i)

    best_j = min(range(len(dp[-1])), key=lambda j: dp[-1][j])
    result = [None]*n; j = best_j
    for i in range(n-1, -1, -1):
        result[i] = candidates[i][j]; j = backptr[i][j]
    return result


def main():
    print("=== Biomechanical Viterbi LOPO ===\n")
    print("Building per-player dataset...")
    player_data, player_eval = build_dataset_by_player()

    players = sorted(player_data.keys())
    print(f"Players: {players}")
    for p in players:
        n_train = len(player_data[p])
        n_eval = sum(len(fd["notes"]) for fd in player_eval.get(p, []))
        print(f"  {p}: {n_train} train, {n_eval} eval notes")

    results_cnn = []; results_bio = []

    for val_player in players:
        train_features = []
        for p in players:
            if p != val_player:
                train_features.extend(player_data[p])

        print(f"\nFold: val={val_player} (train={len(train_features)})")
        model, device = train_cnn(train_features, epochs=50)

        eval_files = player_eval.get(val_player, [])
        if not eval_files:
            print(f"  No solo files for player {val_player}")
            continue

        cnn_correct = 0; cnn_total = 0
        bio_correct = 0; bio_total = 0

        for fd in eval_files:
            notes = fd["notes"]
            # Get CNN probs for each note
            cnn_probs_list = []
            for no in notes:
                probs = predict_probs_with_model(model, device, no["patch"], no["pitch"])
                cnn_probs_list.append(probs)

                # CNN-only accuracy
                possible = []
                for si, op in enumerate(STANDARD_TUNING):
                    s = 6 - si; f = no["pitch"] - op
                    if 0 <= f <= 19:
                        possible.append((s, probs.get(s, 0)))
                if possible:
                    best_s = max(possible, key=lambda x: x[1])[0]
                    cnn_total += 1
                    if best_s == no["gt_string"]:
                        cnn_correct += 1

            # Biomechanical Viterbi
            result = viterbi_bio(notes, cnn_probs_list)
            for i, (ps, pf, pfig) in enumerate(result):
                bio_total += 1
                if ps == notes[i]["gt_string"]:
                    bio_correct += 1

        cnn_acc = cnn_correct / max(cnn_total, 1) * 100
        bio_acc = bio_correct / max(bio_total, 1) * 100
        print(f"  CNN only: {cnn_correct}/{cnn_total} = {cnn_acc:.1f}%")
        print(f"  Bio Viterbi: {bio_correct}/{bio_total} = {bio_acc:.1f}%")
        results_cnn.append((val_player, cnn_correct, cnn_total))
        results_bio.append((val_player, bio_correct, bio_total))

    print("\n=== LOPO Summary ===")
    print(f"{'Player':>8s} {'CNN':>8s} {'Bio':>8s} {'Δ':>6s}")
    tc_cnn = 0; tt_cnn = 0; tc_bio = 0; tt_bio = 0
    for (p, cc, ct), (_, bc, bt) in zip(results_cnn, results_bio):
        ca = cc/max(ct,1)*100; ba = bc/max(bt,1)*100
        delta = ba - ca
        print(f"  {p:>6s} {ca:7.1f}% {ba:7.1f}% {delta:+5.1f}%")
        tc_cnn += cc; tt_cnn += ct; tc_bio += bc; tt_bio += bt

    oa_cnn = tc_cnn / max(tt_cnn, 1) * 100
    oa_bio = tc_bio / max(tt_bio, 1) * 100
    print(f"\n  Overall CNN LOPO:  {tc_cnn}/{tt_cnn} = {oa_cnn:.1f}%")
    print(f"  Overall Bio LOPO:  {tc_bio}/{tt_bio} = {oa_bio:.1f}%")
    print(f"  Improvement: {oa_bio - oa_cnn:+.1f}%")


if __name__ == "__main__":
    main()
