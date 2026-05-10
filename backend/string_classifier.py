"""
弦分類器 (String Classifier)

GuitarSetのhexピックアップ（6ch = 6弦独立信号）を正解データとし、
mono-mic音源のCQT特徴量から「どの弦で弾かれたか」を推定するモデル。

理論的背景:
- ピッチだけでは弦割り当て精度の理論的上限は~70%
- 各弦は異なる倍音構造を持ち、スペクトル特徴で弦を識別可能
- GuitarSetのhexピックアップが各弦の独立信号を提供

アーキテクチャ: 軽量CNN (CQT patch → 6-class softmax)
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jams
import soundfile as sf

# --- 定数 ---
from solotab_utils import STANDARD_TUNING
SR = 22050  # ダウンサンプリング先
HOP_LENGTH = 512
N_BINS = 84  # CQTのビン数
CONTEXT_FRAMES = 11  # オンセット前後の時間的文脈 (±5フレーム)

# GuitarSetのディレクトリ
GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATION_DIR = os.path.join(GUITARSET_DIR, "annotation")
HEX_AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_hex-pickup_debleeded")
MIC_AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")


class StringClassifierCNN(nn.Module):
    """
    軽量CNN弦分類器。
    入力: CQT patch (1, N_BINS, CONTEXT_FRAMES) + pitch (1,)
    出力: 6弦の確率 (6,)
    """
    def __init__(self, n_bins=N_BINS, n_frames=CONTEXT_FRAMES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 + 1, 128),  # +1 for pitch
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6),  # 6弦
        )
    
    def forward(self, cqt_patch, pitch):
        """
        cqt_patch: (batch, 1, n_bins, n_frames)
        pitch: (batch, 1)  - normalized MIDI pitch
        """
        x = self.conv(cqt_patch)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, pitch], dim=1)
        return self.fc(x)


def compute_cqt(audio_path: str) -> np.ndarray:
    """mono-mic音源のCQTスペクトログラムを計算。"""
    import librosa
    y, sr_orig = sf.read(audio_path)
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # mono化
    
    # リサンプリング
    if sr_orig != SR:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
    
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, 
                              n_bins=N_BINS, bins_per_octave=12))
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt = (cqt + 80) / 80  # 正規化 [0, 1]
    cqt = np.clip(cqt, 0, 1)
    return cqt  # (N_BINS, T)


def get_string_from_hex(hex_audio_path: str, onset_time: float,
                         duration: float = 0.05) -> int:
    """
    hexピックアップ音源（6ch）から、オンセット時刻での最大エネルギー弦を特定。
    
    Returns: string_num (1-6), 1=高E, 6=低E
    """
    data, sr_orig = sf.read(hex_audio_path)
    if data.shape[1] != 6:
        return -1
    
    start_sample = int(onset_time * sr_orig)
    end_sample = int((onset_time + duration) * sr_orig)
    start_sample = max(0, start_sample)
    end_sample = min(data.shape[0], end_sample)
    
    if start_sample >= end_sample:
        return -1
    
    # 各チャネルのRMSエネルギー
    segment = data[start_sample:end_sample, :]
    energies = np.sqrt(np.mean(segment ** 2, axis=0))
    
    # GuitarSetのhexチャネル順: ch0=低E(6弦), ch5=高E(1弦)
    best_ch = np.argmax(energies)
    string_num = 6 - best_ch  # ch0→6弦, ch5→1弦
    
    return string_num


def build_dataset(max_tracks: int = 360):
    """
    GuitarSetから学習データセットを構築。
    
    各ノートについて:
    - 入力: mono-mic CQTのオンセット周辺パッチ + ピッチ
    - 正解: hexピックアップから特定した弦番号
    """
    print("=== 弦分類器データセット構築 ===")
    
    jams_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.jams")))[:max_tracks]
    
    all_features = []  # (cqt_patch, pitch, string_label)
    
    for jams_path in jams_files:
        basename = os.path.basename(jams_path).replace(".jams", "")
        
        # 対応するmono-mic音源
        mic_path = os.path.join(MIC_AUDIO_DIR, basename + "_mic.wav")
        if not os.path.exists(mic_path):
            continue
        
        # 対応するhexピックアップ音源
        hex_name = basename + "_hex_cln.wav"
        hex_path = os.path.join(HEX_AUDIO_DIR, hex_name)
        if not os.path.exists(hex_path):
            continue
        
        # CQT計算
        try:
            cqt = compute_cqt(mic_path)
        except Exception as e:
            print(f"  Skip {basename}: CQT error: {e}")
            continue
        
        # JAMS読み込み
        try:
            jam = jams.load(jams_path)
        except Exception:
            continue
        
        # note_midiアノテーションからノート情報を取得
        note_midi_idx = 0
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue
            
            string_num = 6 - note_midi_idx
            note_midi_idx += 1
            if string_num < 1 or string_num > 6:
                continue
            
            for obs in ann.data:
                onset_time = float(obs.time)
                midi_pitch = int(round(obs.value))
                
                # CQTフレームインデックス
                frame_idx = int(onset_time * SR / HOP_LENGTH)
                half_ctx = CONTEXT_FRAMES // 2
                
                if frame_idx - half_ctx < 0 or frame_idx + half_ctx >= cqt.shape[1]:
                    continue
                
                # CQTパッチ抽出
                patch = cqt[:, frame_idx - half_ctx:frame_idx + half_ctx + 1]
                
                # hexピックアップで弦確認（アノテーションの弦とhexの弦が一致するか）
                hex_string = get_string_from_hex(hex_path, onset_time)
                
                # アノテーション弦をGT、hexを補強情報として使用
                gt_string = string_num  # JMASアノテーションの弦順序を信頼
                
                all_features.append({
                    'patch': patch.astype(np.float32),
                    'pitch': midi_pitch,
                    'string': gt_string,  # 1-6
                })
        
        print(f"  {basename}: {len(all_features)} samples累計")
    
    print(f"\n合計サンプル数: {len(all_features)}")
    return all_features


class StringDataset(Dataset):
    def __init__(self, features, augment=False):
        self.features = features
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        f = self.features[idx]
        patch = f['patch'].copy()  # (N_BINS, CONTEXT_FRAMES)
        pitch = f['pitch']
        
        if self.augment:
            # 1. 周波数軸シフト（±2ビン）- 異なるギターの倍音構成をシミュレート
            shift = np.random.randint(-2, 3)
            if shift != 0:
                patch = np.roll(patch, shift, axis=0)
                if shift > 0:
                    patch[:shift, :] = 0
                else:
                    patch[shift:, :] = 0
            
            # 2. 時間軸ジッター（±1フレーム）
            t_shift = np.random.randint(-1, 2)
            if t_shift != 0:
                patch = np.roll(patch, t_shift, axis=1)
                if t_shift > 0:
                    patch[:, :t_shift] = 0
                else:
                    patch[:, t_shift:] = 0
            
            # 3. ガウスノイズ
            noise = np.random.randn(*patch.shape).astype(np.float32) * 0.02
            patch = patch + noise
            
            # 4. ゲイン変動（±20%）
            gain = 1.0 + np.random.uniform(-0.2, 0.2)
            patch = patch * gain
            
            patch = np.clip(patch, 0, 1)
        
        patch_tensor = torch.FloatTensor(patch).unsqueeze(0)  # (1, N_BINS, CONTEXT_FRAMES)
        pitch_tensor = torch.FloatTensor([(pitch - 40) / 45.0])  # 正規化
        label = f['string'] - 1  # 0-5
        return patch_tensor, pitch_tensor, label


def train_classifier(features, epochs=30, batch_size=64, val_split=0.2):
    """弦分類器を学習。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Train/Val分割
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    val_size = int(len(features) * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_features = [features[i] for i in train_idx]
    val_features = [features[i] for i in val_idx]
    
    train_loader = DataLoader(StringDataset(train_features, augment=True), 
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(StringDataset(val_features, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = StringClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for patches, pitches, labels in train_loader:
            patches = patches.to(device)
            pitches = pitches.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(patches, pitches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for patches, pitches, labels in val_loader:
                patches = patches.to(device)
                pitches = pitches.to(device)
                labels = labels.to(device)
                
                outputs = model(patches, pitches)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / train_total
        scheduler.step(avg_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            marker = " ★"
        else:
            marker = ""
        
        print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}{marker}")
    
    # ベストモデル保存
    if best_model_state:
        model.load_state_dict(best_model_state)
        save_path = os.path.join(os.path.dirname(__file__), "string_classifier.pth")
        torch.save(best_model_state, save_path)
        print(f"\nBest val accuracy: {best_val_acc:.4f}")
        print(f"Model saved: {save_path}")
    
    return model, best_val_acc


def predict_string_probs(model, cqt: np.ndarray, onset_time: float,
                          midi_pitch: int, device=None) -> dict:
    """
    学習済みモデルで弦確率を予測。
    
    Returns: {1: prob, 2: prob, ..., 6: prob}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    frame_idx = int(onset_time * SR / HOP_LENGTH)
    half_ctx = CONTEXT_FRAMES // 2
    
    if frame_idx - half_ctx < 0 or frame_idx + half_ctx >= cqt.shape[1]:
        return {}
    
    patch = cqt[:, frame_idx - half_ctx:frame_idx + half_ctx + 1]
    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
    pitch_tensor = torch.FloatTensor([(midi_pitch - 40) / 45.0]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(patch_tensor, pitch_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    return {s+1: float(probs[s]) for s in range(6)}


if __name__ == "__main__":
    print("Phase 1: データセット構築...")
    features = build_dataset(max_tracks=360)
    
    if len(features) < 100:
        print("データが少なすぎます。")
    else:
        print(f"\nPhase 2: モデル学習 ({len(features)} samples)...")
        model, val_acc = train_classifier(features, epochs=100, batch_size=64)
        print(f"\n=== 完了 ===")
        print(f"弦分類器 Val精度: {val_acc:.4f} ({val_acc*100:.2f}%)")
