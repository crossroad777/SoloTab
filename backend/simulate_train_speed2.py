import time
import torch
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from torch.utils.data import DataLoader, Dataset
from data_processing.batching import collate_fn_pad
from training import loss_functions, epoch_processing
from model import architecture
import config

class DummyFastDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.features = torch.randn(3, 192, 430)
        self.onset = torch.zeros(430, 6)
        self.fret = torch.full((430, 6), 22, dtype=torch.long)
        self.raw_labels = torch.tensor([[0.5, 1.0, 1.0, 5.0, 0.0]])
        self.path = "dummy_path"
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.features.clone(), (self.onset.clone(), self.fret.clone()), self.raw_labels.clone(), self.path

def run_verify_speed():
    print("=" * 60)
    print(" 🚀 再検証テスト: DataLoader & Eval Metrics Speed Profiler ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1000 items represents ~1/3 of an epoch
    dataset = DummyFastDataset(size=1000)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=False, collate_fn=collate_fn_pad, num_workers=0
    )

    print("\n初期化中...")
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            batch = x.shape[0]
            frames = x.shape[-1]
            return torch.randn(batch, frames, 6).to(x.device), torch.randn(batch, frames, 6, 23).to(x.device)

    model = MockModel().to(device)
    
    criterion = loss_functions.CombinedLoss(onset_loss_weight=1.0).to(device)

    print(f"\n[計測開始] 1000サンプルのValidationを実行中... (バッチサイズ: {config.BATCH_SIZE_DEFAULT})")
    start_time = time.time()
    
    class DummyPbar(list):
        def set_postfix(self, *args, **kwargs): pass
    pbar = DummyPbar(loader)
    
    val_metrics = epoch_processing.evaluate_one_epoch(model, pbar, criterion, device, config)
    
    end_time = time.time()
    dur = end_time - start_time
    print(f"\n✅ 完了! 1000サンプルの検証処理にかかった時間: {dur:.4f} 秒")
    print(f"   => バッチあたり: {dur / len(loader):.4f} 秒, 1サンプルあたり: {dur / 1000:.4f} 秒")
    print(f"計算された指標一覧: {list(val_metrics.keys())[:5]} ... etc")
    print("=" * 60)
    
if __name__ == "__main__":
    run_verify_speed()
