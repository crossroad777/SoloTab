"""
fingering_model_v3.py — V3 FingeringTransformer モデル定義
==========================================================
GProTab 800万ノートで学習した記号ベース運指予測Transformer。

入力:
  - 文脈窓 (ctx_len=16): pitch, string, fret, duration, interval
  - ターゲット: pitch, duration, interval, position_context

出力:
  - 6弦の確率 (softmax)

重みファイルの構造から逆算して再構築。
元ファイルが紛失したため、保存済み state_dict のキー/shape から
正確にアーキテクチャを復元したもの。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """学習済み位置エンコーディング (最大32トークン)。"""
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        # pe: (1, max_len, d_model) — 重みファイルに保存されているバッファ
        self.register_buffer('pe', torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class FingeringTransformer(nn.Module):
    """
    V3 FingeringTransformer — 記号ベース運指予測。

    Parameters
    ----------
    d_model : int
        Transformer の隠れ次元 (default: 192)
    nhead : int
        Multi-head attention のヘッド数 (default: 6)
    num_layers : int
        TransformerEncoderLayer の層数 (default: 4)
    embed_dim : int
        各特徴量の embedding 次元 (default: 48)
    dropout : float
        Dropout 率 (default: 0.1)
    """
    def __init__(self, d_model=192, nhead=6, num_layers=4,
                 embed_dim=48, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.embed_dim = embed_dim

        # --- Embedding layers ---
        # 各特徴量を独立にembedding
        # pitch: MIDI 0-127 → embed_dim (48)
        self.pitch_emb = nn.Embedding(128, embed_dim)
        # string: 0-6 (0=padding, 1-6=弦) → embed_dim//2 (24)
        self.string_emb = nn.Embedding(7, embed_dim // 2)
        # fret: 0-24 → embed_dim//2 (24)
        self.fret_emb = nn.Embedding(25, embed_dim // 2)
        # duration: 0-31 (量子化済み) → embed_dim//4 (12)
        self.duration_emb = nn.Embedding(32, embed_dim // 4)
        # interval: 0-48 (ピッチ差 -24~+24 → 0~48) → embed_dim//4 (12)
        self.interval_emb = nn.Embedding(49, embed_dim // 4)

        # 各トークンの特徴量を連結: 48 + 24 + 24 + 12 + 12 = 120
        input_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim // 4 + embed_dim // 4
        self.input_proj = nn.Linear(input_dim, d_model)

        # --- Position context embedding ---
        # 直近ノートのフレット中央値 (0-24) → embed_dim (48)
        self.pos_ctx_emb = nn.Embedding(25, embed_dim)

        # --- Positional encoding ---
        self.pos_encoding = PositionalEncoding(d_model, max_len=32)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 768
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --- Target pitch projection ---
        # ターゲットのpitch embedding → d_model
        self.target_pitch_proj = nn.Linear(embed_dim, d_model)

        # --- Output head ---
        # 入力: transformer_output (d_model) + target_pitch_proj (d_model) + pos_ctx (embed_dim)
        # = 192 + 192 + 48 = 432
        self.output_head = nn.Sequential(
            nn.Linear(d_model + d_model + embed_dim, d_model),  # 432 → 192
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),                   # 192 → 96
            nn.ReLU(),
            nn.Linear(d_model // 2, 6),                          # 96 → 6
        )

    def forward(self, ctx_pitches, ctx_strings, ctx_frets,
                ctx_durations, ctx_intervals,
                target_pitch, target_duration, target_interval,
                position_context):
        """
        Parameters
        ----------
        ctx_pitches : (batch, ctx_len) — 文脈窓のMIDIピッチ
        ctx_strings : (batch, ctx_len) — 文脈窓の弦番号 (0-6)
        ctx_frets : (batch, ctx_len) — 文脈窓のフレット (0-24)
        ctx_durations : (batch, ctx_len) — 文脈窓の持続時間 (量子化, 0-31)
        ctx_intervals : (batch, ctx_len) — 文脈窓のピッチ差 (0-48)
        target_pitch : (batch,) — ターゲットのMIDIピッチ
        target_duration : (batch,) — ターゲットの持続時間
        target_interval : (batch,) — ターゲットのピッチ差
        position_context : (batch,) — ポジション文脈 (フレット中央値, 0-24)

        Returns
        -------
        logits : (batch, 6) — 6弦の未正規化スコア
        """
        # Embed context tokens
        p_emb = self.pitch_emb(ctx_pitches)          # (B, L, 48)
        s_emb = self.string_emb(ctx_strings)         # (B, L, 24)
        f_emb = self.fret_emb(ctx_frets)             # (B, L, 24)
        d_emb = self.duration_emb(ctx_durations)     # (B, L, 12)
        i_emb = self.interval_emb(ctx_intervals)     # (B, L, 12)

        # Concatenate: (B, L, 120)
        token_features = torch.cat([p_emb, s_emb, f_emb, d_emb, i_emb], dim=-1)

        # Project to d_model: (B, L, 192)
        x = self.input_proj(token_features)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder: (B, L, 192)
        x = self.transformer(x)

        # Pool: 最終トークンの出力を使用 (autoregressive style)
        ctx_out = x[:, -1, :]  # (B, 192)

        # Target pitch embedding + projection
        tp_emb = self.pitch_emb(target_pitch)         # (B, 48)
        tp_proj = self.target_pitch_proj(tp_emb)      # (B, 192)

        # Position context embedding
        pc_emb = self.pos_ctx_emb(position_context)   # (B, 48)

        # Concatenate: (B, 432)
        combined = torch.cat([ctx_out, tp_proj, pc_emb], dim=-1)

        # Output head: (B, 6)
        logits = self.output_head(combined)

        return logits
