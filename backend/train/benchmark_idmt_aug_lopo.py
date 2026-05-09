"""
CNN LOPO with IDMT + Enhanced Augmentation
==========================================
1. GuitarSet LOPO + IDMT-SMTデータ追加（全fold共通の外部データ）
2. 強化Data Augmentation（pitch shift, gain variation, mixup等）
3. Bio Viterbiで最終精度測定
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jams
from collections import Counter

from string_classifier import (
    StringClassifierCNN, compute_cqt,
    STANDARD_TUNING, SR, HOP_LENGTH, CONTEXT_FRAMES, N_BINS,
    ANNOTATION_DIR, MIC_AUDIO_DIR
)
from extract_idmt_features import extract_idmt_features

MAX_FRET = 19
FINGER_EASE = {0: 0.40, 1: 0.35, 2: 0.30, 3: 0.25, 4: 0.10}
MAX_SPAN = {(1,2):4, (1,3):5, (1,4):6, (2,3):3, (2,4):4, (3,4):3}
W_STRING = 0.3; W_POS = 0.5; W_EASE = 0.5; W_OPEN = 1.0


class EnhancedStringDataset(Dataset):
    """強化版Data Augmentation"""
    def __init__(self, features, augment=False):
        self.features = features
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        patch = f['patch'].copy()
        pitch = f['pitch']

        if self.augment:
            # 1. 周波数シフト（±3ビン、v1より広い）
            shift = np.random.randint(-3, 4)
            if shift != 0:
                patch = np.roll(patch, shift, axis=0)
                if shift > 0: patch[:shift, :] = 0
                else: patch[shift:, :] = 0

            # 2. 時間ジッター（±2フレーム、v1より広い）
            t_shift = np.random.randint(-2, 3)
            if t_shift != 0:
                patch = np.roll(patch, t_shift, axis=1)
                if t_shift > 0: patch[:, :t_shift] = 0
                else: patch[:, t_shift:] = 0

            # 3. ガウスノイズ（強化）
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*patch.shape).astype(np.float32) * noise_level
            patch = patch + noise

            # 4. ゲイン変動（±30%、v1より広い）
            gain = 1.0 + np.random.uniform(-0.3, 0.3)
            patch = patch * gain

            # 5. 周波数マスキング（SpecAugment風）
            if np.random.random() < 0.3:
                f_start = np.random.randint(0, max(1, patch.shape[0] - 5))
                f_width = np.random.randint(1, 6)
                patch[f_start:f_start+f_width, :] = 0

            # 6. 時間マスキング
            if np.random.random() < 0.3:
                t_start = np.random.randint(0, max(1, patch.shape[1] - 3))
                t_width = np.random.randint(1, 4)
                patch[:, t_start:t_start+t_width] = 0

            patch = np.clip(patch, 0, 1)

        patch_tensor = torch.FloatTensor(patch).unsqueeze(0)
        pitch_tensor = torch.FloatTensor([(pitch - 40) / 45.0])
        label = f['string'] - 1
        return patch_tensor, pitch_tensor, label


def build_dataset_by_player():
    jams_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.jams")))
    player_data = {}; player_eval = {}

    for jf in jams_files:
        basename = os.path.basename(jf).replace(".jams", "")
        player = basename[:2]
        mic_path = os.path.join(MIC_AUDIO_DIR, basename + "_mic.wav")
        if not os.path.exists(mic_path): continue
        is_solo = "_solo" in basename

        try:
            cqt = compute_cqt(mic_path); jam = jams.load(jf)
        except: continue

        idx = 0
        for ann in jam.annotations:
            if ann.namespace != "note_midi": continue
            sn = 6 - idx; idx += 1
            if sn < 1 or sn > 6: continue
            si = 6 - sn
            for obs in ann.data:
                onset = float(obs.time); pitch = int(round(obs.value))
                fret = pitch - STANDARD_TUNING[si]
                frame_idx = int(onset * SR / HOP_LENGTH)
                half_ctx = CONTEXT_FRAMES // 2
                if frame_idx - half_ctx < 0 or frame_idx + half_ctx >= cqt.shape[1]: continue
                patch = cqt[:, frame_idx - half_ctx:frame_idx + half_ctx + 1]
                if player not in player_data: player_data[player] = []
                player_data[player].append({"patch": patch.astype(np.float32), "pitch": pitch, "string": sn})

                if is_solo and 0 <= fret <= MAX_FRET:
                    if player not in player_eval: player_eval[player] = []
                    # Flatten: just collect notes with player tag
                    player_eval[player].append({
                        "pitch": pitch, "start": onset, "gt_string": sn, "gt_fret": fret,
                        "patch": patch.astype(np.float32), "basename": basename
                    })

    return player_data, player_eval


def train_cnn(train_features, epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(EnhancedStringDataset(train_features, augment=True),
                        batch_size=batch_size, shuffle=True, num_workers=0)
    model = StringClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = 1e9; best_state = None
    for epoch in range(epochs):
        model.train(); total_loss = 0; total_n = 0
        for patches, pitches, labels in loader:
            patches, pitches, labels = patches.to(device), pitches.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(patches, pitches)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0); total_n += labels.size(0)
        avg = total_loss / total_n
        scheduler.step(avg)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    return model, device


def predict_probs(model, device, patch, pitch):
    with torch.no_grad():
        p = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
        pn = torch.FloatTensor([[(pitch - 40) / 45.0]]).to(device)
        out = model(p, pn)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return {s+1: float(probs[s]) for s in range(6)}


def get_candidates(pitch):
    cands = []
    for si, op in enumerate(STANDARD_TUNING):
        s = 6 - si; f = pitch - op
        if 0 <= f <= 19:
            if f == 0: cands.append((s, f, 0))
            else:
                for finger in range(1, 5): cands.append((s, f, finger))
    return cands


def transition_cost(s1, f1, fig1, s2, f2, fig2, dt):
    cost = W_STRING * abs(s1 - s2)
    if fig1 == 0 or fig2 == 0:
        if fig2 == 0: cost -= W_OPEN
        return cost
    pos1 = f1-(fig1-1); pos2 = f2-(fig2-1); ps = abs(pos2-pos1)
    if ps > 0:
        tf = 1.0 / max(dt, 0.05)
        cost += W_POS * ps * min(tf, 5.0)
    if fig1 == fig2 and f1 != f2: cost += 2.0
    if s1 == s2:
        if (fig2>fig1 and f2<f1) or (fig2<fig1 and f2>f1): cost += 10.0
    if fig1!=fig2 and fig1>0 and fig2>0:
        span = abs(f2-f1); pair = (min(fig1,fig2),max(fig1,fig2))
        ms = MAX_SPAN.get(pair, 6)
        if span > ms: cost += 1.0*(span-ms)
    return cost


def viterbi_bio(notes, cnn_probs_list):
    n = len(notes)
    if n == 0: return []
    candidates = [get_candidates(notes[i]["pitch"]) or [(1,0,0)] for i in range(n)]
    INF = 1e9; dp = []; backptr = []
    probs0 = cnn_probs_list[0]
    costs0 = [-np.log(max(probs0.get(s,1e-10),1e-10)) - W_EASE*FINGER_EASE.get(fig,0.1)
               for (s,f,fig) in candidates[0]]
    dp.append(costs0); backptr.append([-1]*len(candidates[0]))
    for i in range(1, n):
        dt = notes[i]["start"]-notes[i-1]["start"]; pi = cnn_probs_list[i]
        ci = []; bi = []
        for j,(sj,fj,figj) in enumerate(candidates[i]):
            em = -np.log(max(pi.get(sj,1e-10),1e-10)) - W_EASE*FINGER_EASE.get(figj,0.1)
            bc = INF; bp = 0
            for k,(sk,fk,figk) in enumerate(candidates[i-1]):
                c = dp[i-1][k]+transition_cost(sk,fk,figk,sj,fj,figj,dt)+em
                if c < bc: bc=c; bp=k
            ci.append(bc); bi.append(bp)
        dp.append(ci); backptr.append(bi)
    best_j = min(range(len(dp[-1])), key=lambda j: dp[-1][j])
    result = [None]*n; j = best_j
    for i in range(n-1,-1,-1):
        result[i] = candidates[i][j]; j = backptr[i][j]
    return result


def group_notes_by_file(notes_list):
    """ノートをファイル別にグループ化"""
    files = {}
    for n in notes_list:
        bn = n["basename"]
        if bn not in files: files[bn] = []
        files[bn].append(n)
    return files


def main():
    print("=== IDMT + Enhanced Aug LOPO ===\n")
    print("Building GuitarSet per-player dataset...")
    player_data, player_eval = build_dataset_by_player()

    print("Extracting IDMT features...")
    idmt_features = extract_idmt_features()
    print(f"IDMT features: {len(idmt_features)}")

    players = sorted(player_data.keys())
    for p in players:
        n_eval = len(player_eval.get(p, []))
        print(f"  {p}: {len(player_data[p])} train, {n_eval} eval")

    configs = [
        ("GS only",         False, False),
        ("GS + enhanced",   False, True),
        ("GS + IDMT",       True,  False),
        ("GS + IDMT + enh", True,  True),
    ]

    for config_name, use_idmt, use_enhanced in configs:
        print(f"\n--- Config: {config_name} ---")
        tc_cnn = 0; tt_cnn = 0; tc_bio = 0; tt_bio = 0

        for val_player in players:
            train = []
            for p in players:
                if p != val_player:
                    train.extend(player_data[p])
            if use_idmt:
                train.extend(idmt_features)
            print(f"  Fold val={val_player} (train={len(train)})", end="", flush=True)

            model, device = train_cnn(train, epochs=50)

            eval_notes = player_eval.get(val_player, [])
            files = group_notes_by_file(eval_notes)

            cnn_c = 0; cnn_t = 0; bio_c = 0; bio_t = 0
            for bn, notes in files.items():
                notes.sort(key=lambda x: x["start"])
                probs_list = [predict_probs(model, device, n["patch"], n["pitch"]) for n in notes]

                # CNN only
                for i, n in enumerate(notes):
                    possible = []
                    for si, op in enumerate(STANDARD_TUNING):
                        s = 6 - si; f = n["pitch"] - op
                        if 0 <= f <= 19: possible.append((s, probs_list[i].get(s, 0)))
                    if possible:
                        best_s = max(possible, key=lambda x: x[1])[0]
                        cnn_t += 1
                        if best_s == n["gt_string"]: cnn_c += 1

                # Bio Viterbi
                result = viterbi_bio(notes, probs_list)
                for i, (ps, pf, pfig) in enumerate(result):
                    bio_t += 1
                    if ps == notes[i]["gt_string"]: bio_c += 1

            ca = cnn_c/max(cnn_t,1)*100; ba = bio_c/max(bio_t,1)*100
            print(f"  CNN={ca:.1f}% Bio={ba:.1f}%")
            tc_cnn += cnn_c; tt_cnn += cnn_t; tc_bio += bio_c; tt_bio += bio_t

        oa_cnn = tc_cnn/max(tt_cnn,1)*100; oa_bio = tc_bio/max(tt_bio,1)*100
        print(f"  Overall: CNN={oa_cnn:.1f}% Bio={oa_bio:.1f}% (Δ={oa_bio-oa_cnn:+.1f}%)")


if __name__ == "__main__":
    main()
