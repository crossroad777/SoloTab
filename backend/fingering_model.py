"""
fingering_model.py — シーケンスベース弦予測モデル
==================================================
Bidirectional LSTMで前後の音の文脈を考慮し、
各ノートの最適弦(1-6)を予測する。

入力特徴量 (per note):
  - pitch (MIDI, 正規化)
  - duration (秒)
  - inter-onset interval (前の音からの時間差)
  - CNN弦確率 (6次元, あれば)
  → 合計 9次元

出力:
  - 6クラス分類 (弦1-6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import sys
import glob

# =============================================================================
# モデル定義
# =============================================================================

class FingeringLSTM(nn.Module):
    """Bidirectional LSTM for string prediction."""
    
    def __init__(self, input_dim=9, hidden_dim=128, n_layers=2, n_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len, input_dim)
        lengths: (batch,) — 各シーケンスの実際の長さ
        returns: (batch, seq_len, n_classes)
        """
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# =============================================================================
# データセット
# =============================================================================

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4

def _extract_features(notes):
    """ノートリストから特徴量テンソルを作成。"""
    feats = []
    for i, note in enumerate(notes):
        pitch_norm = (note['pitch'] - 40) / 48.0  # E2(40)~E6(88)を0~1に
        duration = min(note.get('duration', 0.5), 5.0) / 5.0  # 最大5秒で正規化
        
        # Inter-onset interval
        if i > 0:
            ioi = note['start'] - notes[i-1]['start']
        else:
            ioi = 0.0
        ioi = min(ioi, 5.0) / 5.0
        
        # CNN弦確率 (6次元)
        cnn_probs = note.get('cnn_string_probs', {})
        cnn_feat = [cnn_probs.get(str(s), 0.0) for s in range(1, 7)]
        
        feat = [pitch_norm, duration, ioi] + cnn_feat
        feats.append(feat)
    
    return torch.tensor(feats, dtype=torch.float32)


def _extract_labels(notes):
    """正解弦ラベルを抽出 (0-indexed: 0=1弦, 5=6弦)"""
    labels = []
    for note in notes:
        string = note.get('string', 1)
        labels.append(string - 1)  # 0-indexed
    return torch.tensor(labels, dtype=torch.long)


class GuitarSetFingeringDataset(Dataset):
    """GuitarSetからシーケンスデータを作成。"""
    
    def __init__(self, annotation_dir, audio_dir=None, max_files=None):
        self.sequences = []
        self._load_from_jams(annotation_dir, max_files)
    
    def _load_from_jams(self, annotation_dir, max_files):
        import jams
        
        jams_files = sorted(glob.glob(os.path.join(annotation_dir, "*.jams")))
        if max_files:
            jams_files = jams_files[:max_files]
        
        print(f"Loading {len(jams_files)} JAMS files for fingering model...")
        
        for jams_path in jams_files:
            try:
                jam = jams.load(jams_path)
            except Exception:
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
                    if fret < 0 or fret > 19:
                        continue
                    
                    notes.append({
                        'start': start,
                        'pitch': midi_pitch,
                        'duration': duration,
                        'string': string_num,
                        'fret': fret,
                    })
            
            if len(notes) < 5:
                continue
            
            # 時間順ソート
            notes.sort(key=lambda n: n['start'])
            
            feats = _extract_features(notes)
            labels = _extract_labels(notes)
            
            self.sequences.append({
                'features': feats,
                'labels': labels,
                'filename': os.path.basename(jams_path),
                'n_notes': len(notes),
            })
        
        print(f"Loaded {len(self.sequences)} sequences, "
              f"total {sum(s['n_notes'] for s in self.sequences)} notes")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]['features'], self.sequences[idx]['labels']


def collate_fn(batch):
    """可変長シーケンスをパディング。"""
    features, labels = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    features_padded = pad_sequence(features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return features_padded, labels_padded, lengths


# =============================================================================
# 学習
# =============================================================================

def train_fingering_model(annotation_dir, output_path, n_epochs=50, lr=0.001,
                          hidden_dim=128, n_layers=2, batch_size=8):
    """GuitarSetで弦予測LSTMを学習。Leave-one-player-out で評価。"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # データセット作成
    dataset = GuitarSetFingeringDataset(annotation_dir)
    
    if len(dataset) == 0:
        print("No data loaded!")
        return
    
    # GuitarSetの6プレイヤーでLOPO分割
    # ファイル名パターン: "00_BN1-129-Eb_comp.jams" → player = "00"
    player_map = {}
    for i, seq in enumerate(dataset.sequences):
        player = seq['filename'][:2]
        if player not in player_map:
            player_map[player] = []
        player_map[player].append(i)
    
    players = sorted(player_map.keys())
    print(f"Players: {players}")
    
    # 全データで学習（最終モデル用）
    # 検証は最後のプレイヤーで実施
    val_player = players[-1]
    val_indices = set(player_map[val_player])
    train_indices = [i for i in range(len(dataset)) if i not in val_indices]
    
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_data)} sequences, Val: {len(val_data)} sequences ({val_player})")
    
    # モデル
    model = FingeringLSTM(input_dim=9, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for feats, labels, lengths in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(feats, lengths)
            
            # Flatten for loss
            logits_flat = logits.view(-1, 6)
            labels_flat = labels.view(-1)
            
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            mask = labels_flat != -100
            preds = logits_flat[mask].argmax(dim=1)
            train_correct += (preds == labels_flat[mask]).sum().item()
            train_total += mask.sum().item()
        
        train_acc = train_correct / max(train_total, 1)
        
        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for feats, labels, lengths in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                
                logits = model(feats, lengths)
                logits_flat = logits.view(-1, 6)
                labels_flat = labels.view(-1)
                
                mask = labels_flat != -100
                preds = logits_flat[mask].argmax(dim=1)
                val_correct += (preds == labels_flat[mask]).sum().item()
                val_total += mask.sum().item()
        
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(1 - val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            marker = " ★"
        else:
            marker = ""
        
        if (epoch + 1) % 5 == 0 or marker:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}{marker}")
    
    # 最良モデルを保存
    if best_state:
        model.load_state_dict(best_state)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': 9,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'val_acc': best_val_acc,
    }, output_path)
    
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved: {output_path}")
    
    return best_val_acc


# =============================================================================
# 推論（パイプライン統合用）
# =============================================================================

_FINGERING_MODEL = None

def load_fingering_model(model_path=None):
    """学習済みLSTMモデルをロード。"""
    global _FINGERING_MODEL
    if _FINGERING_MODEL is not None:
        return _FINGERING_MODEL
    
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "fingering_lstm.pth")
    
    if not os.path.exists(model_path):
        return None
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = FingeringLSTM(
        input_dim=checkpoint.get('input_dim', 9),
        hidden_dim=checkpoint.get('hidden_dim', 128),
        n_layers=checkpoint.get('n_layers', 2),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    _FINGERING_MODEL = {'model': model, 'device': device}
    print(f"[fingering_model] LSTM loaded (val_acc={checkpoint.get('val_acc', '?'):.4f}, device={device})")
    return _FINGERING_MODEL


def predict_strings(notes, tuning=None):
    """
    ノートリストに対して弦予測を行い、string/fretを更新する。
    
    Args:
        notes: List[dict] — start, pitch, (cnn_string_probs) を含むノートリスト
        tuning: List[int] — チューニング（デフォルト: standard）
    
    Returns:
        notes with updated 'string' and 'fret'
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    fm = load_fingering_model()
    if fm is None:
        return notes  # モデルなし → 何もしない
    
    model = fm['model']
    device = fm['device']
    max_fret = 19
    
    # 特徴量作成
    feats = _extract_features(notes).unsqueeze(0).to(device)  # (1, seq_len, 9)
    
    with torch.no_grad():
        logits = model(feats)  # (1, seq_len, 6)
        probs = F.softmax(logits[0], dim=-1)  # (seq_len, 6)
    
    # 各ノートに弦を割り当て（物理的可能性チェック付き）
    for i, note in enumerate(notes):
        pitch = note['pitch']
        note_probs = probs[i].cpu().numpy()  # (6,) — [1弦prob, ..., 6弦prob]
        
        # 物理的に可能なポジションを列挙
        possible = []
        for si, open_pitch in enumerate(tuning):
            s = 6 - si  # 弦番号 (6=low E)
            fret = pitch - open_pitch
            if 0 <= fret <= max_fret:
                possible.append((s, fret, note_probs[s - 1]))
        
        if not possible:
            continue
        
        # LSTM確率 × 物理的可能性でベストを選択
        best = max(possible, key=lambda x: x[2])
        note['string'] = best[0]
        note['fret'] = best[1]
    
    return notes


# =============================================================================
# メイン（学習実行）
# =============================================================================

if __name__ == "__main__":
    annotation_dir = r"D:\Music\Datasets\GuitarSet\annotation"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingering_lstm.pth")
    
    if not os.path.exists(annotation_dir):
        print(f"GuitarSet not found: {annotation_dir}")
        sys.exit(1)
    
    train_fingering_model(
        annotation_dir=annotation_dir,
        output_path=output_path,
        n_epochs=100,
        lr=0.001,
        hidden_dim=128,
        n_layers=2,
        batch_size=8,
    )
