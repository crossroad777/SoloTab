import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import numpy as np
import librosa
import argparse
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from moe_gating_network import GatingNetworkCNN
import config

# 定義された6つのクラス（Expertsの並び順に合わせる）
EXPERTS = [
    "martin_finger",
    "taylor_finger",
    "luthier_finger",
    "martin_pick",
    "taylor_pick",
    "luthier_pick"
]

class RouterDataset(Dataset):
    """
    Gating Network（ルーター）学習用データセット。
    各ギターのFLACから一部を切り抜き、どのエキスパートに属するか（0〜5）をラベルとする。
    """
    def __init__(self, base_dir=r"D:\Music\datasets", samples_per_class=None, augment=True):
        self.sr = config.SAMPLE_RATE
        self.hop_length = config.HOP_LENGTH
        self.augment = augment
        self.tracks = []
        
        for class_idx, exp_name in enumerate(EXPERTS):
            dir_path = os.path.join(base_dir, exp_name)
            print(f"Scanning {dir_path} ...", flush=True)
            files = []
            file_count = 0
            for root_dir, _, filenames in os.walk(dir_path):
                for name in filenames:
                    if name.endswith("_nonoise_mono_body.flac"):
                        files.append(os.path.join(root_dir, name))
                        file_count += 1
                if samples_per_class is not None and len(files) >= samples_per_class * 2:
                    break
            
            print(f"  Found {file_count} valid files in {exp_name}.", flush=True)
            np.random.shuffle(files)
            files_to_use = files[:samples_per_class] if samples_per_class is not None else files
            for f in files_to_use:
                self.tracks.append({
                    "flac": f,
                    "label": class_idx
                })
        print(f"Router Dataset loaded with {len(self.tracks)} tracks.", flush=True)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        y_tensor, sr_load = torchaudio.load(track["flac"])
        if sr_load != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_load, new_freq=self.sr)
            y_tensor = resampler(y_tensor)
        
        y = y_tensor.squeeze(0).numpy()
        
        # 音声からランダムに3秒程度切り抜く
        target_len = self.sr * 3
        if len(y) > target_len:
            start = np.random.randint(0, len(y) - target_len)
            y = y[start:start+target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
            
        # CQT変換 (元々の安定版ロジック)
        cqt_spec = librosa.cqt(
            y=y, sr=self.sr, hop_length=self.hop_length,
            fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
        )
        log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features = torch.tensor(log_cqt_spec, dtype=torch.float32)
        features = features.unsqueeze(0) # (1, time, freq)
        
        return features, torch.tensor(track["label"], dtype=torch.long)

def train_router(epochs=10, batch_size=32, resume=False, start_epoch=1, start_step=0):
    # D:ドライブなどのネットワーク/低速ドライブでの Numba エラー回避策
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.environ.get("TEMP", "."), "numba_cache")
    
    try:
        print(f"Process PID: {os.getpid()}", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}", flush=True)
        
        # モデルの定義
        model = GatingNetworkCNN(in_channels=1, num_experts=6).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # パスの設定
        checkpoint_dir = os.path.join(project_root, "music-transcription", "python", "_processed_guitarset_data", "training_output", "finetuned_router_model")
        latest_path = os.path.join(checkpoint_dir, "latest_model.pth")
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        
        # 自動再開ロジック (Resilience Standard)
        if resume and os.path.exists(state_path) and start_epoch == 1 and start_step == 0:
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                    start_epoch = state.get("epoch", 1)
                    start_step = state.get("step", 0)
                    print(f"Detected previous state: Epoch {start_epoch}, Step {start_step}", flush=True)
            except Exception as e:
                print(f"Error loading state.json: {e}. Falling back to default resume.", flush=True)

        if resume and os.path.exists(latest_path):
            print(f"Loading weights from {latest_path}", flush=True)
            model.load_state_dict(torch.load(latest_path, map_location=device))
            print(f"Resuming from Epoch {start_epoch}, Step {start_step}", flush=True)
        elif resume:
            old_path = os.path.join(checkpoint_dir, "router_model.pth")
            if os.path.exists(old_path):
                print(f"Loading weights from old checkpoint {old_path}", flush=True)
                model.load_state_dict(torch.load(old_path, map_location=device))
            else:
                print(f"Warning: No checkpoint found. Starting from scratch.", flush=True)
        
        def save_everything(m, e, s):
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(m.state_dict(), latest_path)
            with open(state_path, "w") as f_out:
                json.dump({"epoch": e, "step": s}, f_out)
            print(f"  [Checkpoint Saved] Epoch {e}, Step {s}", flush=True)

        print("Initializing Dataset... (Stable CPU loader active)", flush=True)
        dataset = RouterDataset(samples_per_class=None)
        
        all_indices = list(range(len(dataset)))
        np.random.seed(42)
        np.random.shuffle(all_indices)
        
        print(f"Starting stabilized training from Epoch {start_epoch}.", flush=True)
        model.train()
        
        for epoch in range(start_epoch, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            
            if epoch == start_epoch and start_step > 0:
                current_start_idx = start_step * batch_size
                if current_start_idx >= len(all_indices):
                    print(f"Error: start_step {start_step} is out of bounds.", flush=True)
                    return
                curr_indices = all_indices[current_start_idx:]
                print(f"  Fast-skipping to sample index {current_start_idx} (Step {start_step})...", flush=True)
            else:
                curr_indices = all_indices
                
            current_dataset = Subset(dataset, curr_indices)
            loader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            
            print(f"  Data loader for Epoch {epoch} initialized with {len(current_dataset)} samples.", flush=True)
            
            for batch_idx, (features, labels) in enumerate(loader):
                actual_step = (start_step + batch_idx + 1) if epoch == start_epoch else (batch_idx + 1)
                
                features = features.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (actual_step % 10 == 0):
                    print(f"Epoch [{epoch}/{epochs}], Step [{actual_step}/{(len(all_indices)-1)//batch_size + 1}], Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%", flush=True)
                
                # 500ステップごとに中間保存
                if (actual_step % 500 == 0):
                    save_everything(model, epoch, actual_step)
                    
            epoch_acc = 100 * correct / total
            print(f"==> Epoch {epoch} finished. Average Loss: {total_loss/len(loader):.4f}, Accuracy: {epoch_acc:.2f}%", flush=True)
            
            # エポック終了時保存
            save_everything(model, epoch + 1, 0)
            
            start_step = 0
            np.random.seed(42 + epoch)
            np.random.shuffle(all_indices)

    except Exception as e:
        print("\n" + "="*40, flush=True)
        print(f"CRITICAL ERROR in training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training (autodetect if possible)")
    parser.add_argument("--start_epoch", type=int, default=1, help="Force start from epoch")
    parser.add_argument("--start_step", type=int, default=0, help="Force start from step")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    train_router(
        epochs=args.epochs, batch_size=args.batch_size, resume=args.resume, 
        start_epoch=args.start_epoch, start_step=args.start_step
    )
