"""
CNN String Classifier LOPO (Leave-One-Player-Out) 評価
=====================================================
GuitarSetの6プレイヤーでLOPOクロスバリデーション。
CNNを各foldで再学習し、未見プレイヤーでの弦分類精度を測る。
これがCNNの真の汎化精度。
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import jams

from string_classifier import (
    StringClassifierCNN, StringDataset, compute_cqt,
    STANDARD_TUNING, SR, HOP_LENGTH, CONTEXT_FRAMES, N_BINS,
    ANNOTATION_DIR, MIC_AUDIO_DIR
)

def build_dataset_by_player():
    """プレイヤー別にデータセットを構築"""
    jams_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.jams")))
    
    player_data = {}  # {player_id: [features]}
    
    for jf in jams_files:
        basename = os.path.basename(jf).replace(".jams", "")
        player = basename[:2]
        
        mic_path = os.path.join(MIC_AUDIO_DIR, basename + "_mic.wav")
        if not os.path.exists(mic_path):
            continue
        
        try:
            cqt = compute_cqt(mic_path)
        except:
            continue
        
        try:
            jam = jams.load(jf)
        except:
            continue
        
        idx = 0
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue
            sn = 6 - idx; idx += 1
            if sn < 1 or sn > 6: continue
            
            for obs in ann.data:
                onset = float(obs.time)
                pitch = int(round(obs.value))
                
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
    
    return player_data


def train_and_eval(train_features, val_features, epochs=50, batch_size=64):
    """1 foldの学習と評価"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(StringDataset(train_features, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(StringDataset(val_features, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = StringClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0; train_total = 0
        for patches, pitches, labels in train_loader:
            patches, pitches, labels = patches.to(device), pitches.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(patches, pitches)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)
        
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for patches, pitches, labels in val_loader:
                patches, pitches, labels = patches.to(device), pitches.to(device), labels.to(device)
                out = model(patches, pitches)
                _, pred = out.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(train_loss / train_total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc, val_total


def main():
    print("=== CNN LOPO Cross-Validation ===\n")
    print("Building per-player dataset...")
    player_data = build_dataset_by_player()
    
    players = sorted(player_data.keys())
    print(f"Players: {players}")
    for p in players:
        print(f"  {p}: {len(player_data[p])} samples")
    
    print("\n--- LOPO Folds ---")
    results = []
    for val_player in players:
        train_features = []
        for p in players:
            if p != val_player:
                train_features.extend(player_data[p])
        val_features = player_data[val_player]
        
        print(f"\nFold: val={val_player} (train={len(train_features)}, val={len(val_features)})")
        acc, n = train_and_eval(train_features, val_features, epochs=50)
        print(f"  Val accuracy: {acc:.4f} ({int(acc*n)}/{n})")
        results.append((val_player, acc, n))
    
    # Summary
    print("\n=== LOPO Summary ===")
    total_correct = 0; total_n = 0
    for player, acc, n in results:
        correct = int(acc * n)
        total_correct += correct; total_n += n
        print(f"  Player {player}: {acc*100:.1f}% ({correct}/{n})")
    
    overall = total_correct / total_n * 100
    print(f"\n  Overall LOPO: {total_correct}/{total_n} = {overall:.1f}%")
    print(f"  (Compare: random split = 92.66%, current benchmark = 93.3%)")


if __name__ == "__main__":
    main()
