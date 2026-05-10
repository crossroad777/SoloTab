"""
fingering_model_v3.py — V3: Transformer運指予測モデル
====================================================
V2 LSTM → V3 Transformer:
  - Self-Attention で長距離依存性を捉える
  - duration, interval, position_context を追加入力
  - ベストepoch=1問題の解消（より複雑なパターン学習）

Architecture:
  Embeddings: pitch(128) + string(7) + fret(25) + duration(32) + interval(49)
  → concat → Linear → Transformer Encoder → MLP → 6-class
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FingeringTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, 
                 embed_dim=32, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.pitch_emb = nn.Embedding(128, embed_dim)
        self.string_emb = nn.Embedding(7, embed_dim // 2)  # 0-6
        self.fret_emb = nn.Embedding(25, embed_dim // 2)   # 0-24
        self.duration_emb = nn.Embedding(32, embed_dim // 4)  # 0-31
        self.interval_emb = nn.Embedding(49, embed_dim // 4)  # 0-48
        
        # Total input dim = embed_dim + embed_dim/2 + embed_dim/2 + embed_dim/4 + embed_dim/4
        # = embed_dim * 2.5
        input_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim // 4 + embed_dim // 4
        
        # Project to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Position context embedding
        self.pos_ctx_emb = nn.Embedding(25, d_model // 4)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=32)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Target pitch embedding (separate)
        self.target_pitch_proj = nn.Linear(embed_dim, d_model)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2 + d_model // 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),
        )
    
    def forward(self, ctx_pitches, ctx_strings, ctx_frets, 
                ctx_durations, ctx_intervals,
                target_pitch, target_duration, target_interval,
                position_context):
        """
        All context tensors: (batch, seq_len)
        Target tensors: (batch,)
        position_context: (batch,)
        """
        # Embed context
        p_emb = self.pitch_emb(ctx_pitches)
        s_emb = self.string_emb(ctx_strings)
        f_emb = self.fret_emb(ctx_frets)
        d_emb = self.duration_emb(ctx_durations)
        i_emb = self.interval_emb(ctx_intervals)
        
        # Concat all features
        x = torch.cat([p_emb, s_emb, f_emb, d_emb, i_emb], dim=-1)
        x = self.input_proj(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encode
        x = self.transformer(x)  # (batch, seq, d_model)
        
        # Use last position as context representation
        ctx_repr = x[:, -1]  # (batch, d_model)
        
        # Target pitch representation
        tgt_repr = self.target_pitch_proj(self.pitch_emb(target_pitch))  # (batch, d_model)
        
        # Position context
        pos_repr = self.pos_ctx_emb(position_context)  # (batch, d_model//4)
        
        # Combine and predict
        combined = torch.cat([ctx_repr, tgt_repr, pos_repr], dim=-1)
        logits = self.output_head(combined)
        
        return logits


# V2互換のLSTM (特徴量追加版)
class FingeringLSTMv3(nn.Module):
    """V3特徴量対応のLSTM（Transformerとの比較用）"""
    def __init__(self, embed_dim=48, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.pitch_emb = nn.Embedding(128, embed_dim)
        self.string_emb = nn.Embedding(7, embed_dim // 2)
        self.fret_emb = nn.Embedding(25, embed_dim // 2)
        self.duration_emb = nn.Embedding(32, embed_dim // 4)
        self.interval_emb = nn.Embedding(49, embed_dim // 4)
        
        input_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim // 4 + embed_dim // 4
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        
        self.pos_ctx_emb = nn.Embedding(25, hidden_dim // 4)
        self.target_proj = nn.Linear(embed_dim, hidden_dim)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )
    
    def forward(self, ctx_pitches, ctx_strings, ctx_frets,
                ctx_durations, ctx_intervals,
                target_pitch, target_duration, target_interval,
                position_context):
        p = self.pitch_emb(ctx_pitches)
        s = self.string_emb(ctx_strings)
        f = self.fret_emb(ctx_frets)
        d = self.duration_emb(ctx_durations)
        i = self.interval_emb(ctx_intervals)
        
        x = torch.cat([p, s, f, d, i], dim=-1)
        out, _ = self.lstm(x)
        ctx_repr = out[:, -1]
        
        tgt = self.target_proj(self.pitch_emb(target_pitch))
        pos = self.pos_ctx_emb(position_context)
        
        combined = torch.cat([ctx_repr, tgt, pos], dim=-1)
        return self.head(combined)
