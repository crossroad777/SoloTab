"""
CNN確率付きLSTM学習: GuitarSetのCNN弦確率を事前計算し、LSTM再学習
"""
import json, sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import jams
from string_assigner import (
    _load_string_classifier, _compute_cqt_cached, _predict_string_probs,
    STANDARD_TUNING, MAX_FRET
)
from fingering_model import (
    FingeringLSTM, collate_fn, STANDARD_TUNING as FT,
    _extract_labels
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

ANNOTATION_DIR = r"D:\Music\Datasets\GuitarSet\annotation"
AUDIO_DIR = r"D:\Music\Datasets\GuitarSet\audio_mono-mic"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingering_lstm.pth")


def extract_features_with_cnn(notes):
    """CNN確率付き特徴量 (9次元)"""
    feats = []
    for i, note in enumerate(notes):
        pitch_norm = (note['pitch'] - 40) / 48.0
        duration = min(note.get('duration', 0.5), 5.0) / 5.0
        if i > 0:
            ioi = note['start'] - notes[i-1]['start']
        else:
            ioi = 0.0
        ioi = min(ioi, 5.0) / 5.0
        
        cnn_probs = note.get('cnn_string_probs', {})
        cnn_feat = [cnn_probs.get(str(s), cnn_probs.get(s, 0.0)) for s in range(1, 7)]
        
        feat = [pitch_norm, duration, ioi] + cnn_feat
        feats.append(feat)
    return torch.tensor(feats, dtype=torch.float32)


def load_guitarset_with_cnn():
    """GuitarSetの全トラックにCNN弦確率を付与して読み込む"""
    
    # CNN弦分類器をロード
    clf = _load_string_classifier()
    if not clf:
        print("ERROR: CNN string classifier not found!")
        sys.exit(1)
    
    jams_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.jams")))
    print(f"Loading {len(jams_files)} JAMS files with CNN probs...")
    
    sequences = []
    total_notes = 0
    cnn_injected = 0
    
    for ji, jams_path in enumerate(jams_files):
        basename = os.path.splitext(os.path.basename(jams_path))[0]
        
        # 対応する音声ファイルを探す
        wav_candidates = [
            os.path.join(AUDIO_DIR, basename + ".wav"),
            os.path.join(AUDIO_DIR, basename + "_mic.wav"),
        ]
        audio_path = None
        for c in wav_candidates:
            if os.path.exists(c):
                audio_path = c
                break
        
        if audio_path is None:
            # ファイル名のマッチングを試みる
            # JAMS: "00_BN1-129-Eb_solo.jams" -> audio: "00_BN1-129-Eb_solo_mic.wav"
            wav_glob = glob.glob(os.path.join(AUDIO_DIR, basename + "*"))
            if wav_glob:
                audio_path = wav_glob[0]
        
        try:
            jam = jams.load(jams_path)
        except:
            continue
        
        notes = []
        note_midi_idx = 0
        
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue
            string_num = 6 - note_midi_idx
            note_midi_idx += 1
            if string_num < 1 or string_num > 6:
                continue
            string_idx = 6 - string_num
            
            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                start = float(obs.time)
                duration = float(obs.duration)
                fret = midi_pitch - STANDARD_TUNING[string_idx]
                if fret < 0 or fret > MAX_FRET:
                    continue
                
                note = {
                    'start': start,
                    'pitch': midi_pitch,
                    'duration': duration,
                    'string': string_num,
                    'fret': fret,
                }
                
                # CNN弦確率を注入
                if audio_path:
                    probs = _predict_string_probs(audio_path, start, midi_pitch)
                    if probs:
                        note['cnn_string_probs'] = probs
                        cnn_injected += 1
                
                notes.append(note)
        
        if len(notes) < 5:
            continue
        
        notes.sort(key=lambda n: n['start'])
        total_notes += len(notes)
        
        feats = extract_features_with_cnn(notes)
        labels = _extract_labels(notes)
        
        sequences.append({
            'features': feats,
            'labels': labels,
            'filename': os.path.basename(jams_path),
            'n_notes': len(notes),
        })
        
        if (ji + 1) % 30 == 0:
            print(f"  {ji+1}/{len(jams_files)} processed ({total_notes} notes, {cnn_injected} CNN-injected)")
    
    print(f"Loaded {len(sequences)} sequences, {total_notes} notes, {cnn_injected} CNN-injected")
    return sequences


def train(sequences, n_epochs=100, lr=0.001, hidden_dim=128, n_layers=2, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # LOPO split
    player_map = {}
    for i, seq in enumerate(sequences):
        player = seq['filename'][:2]
        if player not in player_map:
            player_map[player] = []
        player_map[player].append(i)
    
    players = sorted(player_map.keys())
    val_player = players[-1]
    val_indices = set(player_map[val_player])
    
    train_data = [(sequences[i]['features'], sequences[i]['labels']) for i in range(len(sequences)) if i not in val_indices]
    val_data = [(sequences[i]['features'], sequences[i]['labels']) for i in val_indices]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)} ({val_player})")
    
    model = FingeringLSTM(input_dim=9, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        for feats, labels, lengths in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats, lengths)
            loss = criterion(logits.view(-1, 6), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            mask = labels.view(-1) != -100
            preds = logits.view(-1, 6)[mask].argmax(dim=1)
            train_correct += (preds == labels.view(-1)[mask]).sum().item()
            train_total += mask.sum().item()
        
        train_acc = train_correct / max(train_total, 1)
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels, lengths in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats, lengths)
                mask = labels.view(-1) != -100
                preds = logits.view(-1, 6)[mask].argmax(dim=1)
                val_correct += (preds == labels.view(-1)[mask]).sum().item()
                val_total += mask.sum().item()
        
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(1 - val_acc)
        
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★"
        
        if (epoch + 1) % 5 == 0 or marker:
            print(f"Epoch {epoch+1:3d}: train={train_acc:.4f} val={val_acc:.4f}{marker}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': 9,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'val_acc': best_val_acc,
        'has_cnn_features': True,
    }, OUTPUT_PATH)
    
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    sequences = load_guitarset_with_cnn()
    train(sequences, n_epochs=25, lr=0.001, hidden_dim=128, n_layers=2)
