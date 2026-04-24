import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetworkCNN(nn.Module):
    """
    動的MoEアプローチ (案D) 用の Gating Network (ルーター).
    入力された音響特徴（CQTやMelスペクトログラム）から、フレームごとに
    6つのギター特化モデル（Experts）への重み（確率）を算出します。
    """
    def __init__(self, in_channels=1, num_experts=6):
        super(GatingNetworkCNN, self).__init__()
        # 軽量・高速な判定を行うためのCNNブロック
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        
        # 入力次元に依存しないAdaptivePooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_experts)
        
    def forward(self, x):
        # x shape: (Batch, Channels, time_frames, freq_bins)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        # 各Expert（特化モデル）への採用割合を0.0~1.0の確率(合計1)として出力
        gating_weights = F.softmax(self.out(x), dim=1)
        return gating_weights
