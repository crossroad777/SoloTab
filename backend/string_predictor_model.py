"""
string_predictor_model.py — Transformer MLMベース弦予測モデル
=============================================================
MIDIピッチ列から最適なギター弦（0-5）を予測する。
ISMIR 2024 "MIDI-to-Tab" (Edwards et al.) のアプローチに基づく。

核心: ピッチが既知なら弦を予測 → フレットは自動計算
    fret = pitch - open_string_pitch[string]

アーキテクチャ: Encoder-only Transformer (BERT-like MLM)
入力: pitch + time_shift + duration
出力: 6クラス分類（弦0-5）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StringPredictor(nn.Module):
    """Transformer-based guitar string predictor using MLM paradigm.
    
    Input features per note:
        - pitch: MIDI note number (0-127)
        - time_shift: quantized inter-onset interval (0-31)
        - duration: quantized note duration (0-31)
    
    Output: string prediction (0-5) for each note position
    
    Training: Masked Language Modeling
        - Randomly mask string labels in training data
        - Model predicts masked strings given pitch context
    
    Inference: Iterative decoding
        - Mask all strings → predict → accept high-confidence → repeat
    """
    
    def __init__(
        self,
        num_strings: int = 6,
        pitch_vocab_size: int = 128,
        time_vocab_size: int = 32,
        duration_vocab_size: int = 32,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        self.num_strings = num_strings
        self.d_model = d_model
        
        # Embeddings
        self.pitch_embed = nn.Embedding(pitch_vocab_size, d_model // 2)
        self.time_embed = nn.Embedding(time_vocab_size, d_model // 4)
        self.duration_embed = nn.Embedding(duration_vocab_size, d_model // 4)
        
        # Project concatenated embeddings to d_model
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output head: string classification
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_strings),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, pitches, time_shifts, durations, 
                padding_mask=None):
        """
        Args:
            pitches: (batch, seq_len) — MIDI pitch indices
            time_shifts: (batch, seq_len) — quantized time shift indices
            durations: (batch, seq_len) — quantized duration indices
            padding_mask: (batch, seq_len) — True where padded
            
        Returns:
            logits: (batch, seq_len, num_strings) — string prediction logits
        """
        # Embed inputs
        pitch_emb = self.pitch_embed(pitches)         # (B, L, d/2)
        time_emb = self.time_embed(time_shifts)        # (B, L, d/4)
        dur_emb = self.duration_embed(durations)       # (B, L, d/4)
        
        # Concatenate and project
        combined = torch.cat([pitch_emb, time_emb, dur_emb], dim=-1)  # (B, L, d)
        x = self.input_proj(combined)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        if padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        
        # Classify strings
        logits = self.output_head(x)  # (B, L, 6)
        
        return logits
    
    def predict_strings(self, pitches, time_shifts, durations,
                        lengths=None, num_iterations: int = 3,
                        confidence_threshold: float = 0.8):
        """Iterative MLM-style inference for string prediction.
        
        Instead of predicting all at once, iteratively:
        1. Predict all strings
        2. Accept predictions with confidence > threshold
        3. Re-predict remaining with accepted context
        
        Args:
            pitches: (batch, seq_len)
            time_shifts: (batch, seq_len) 
            durations: (batch, seq_len)
            lengths: (batch,) — actual sequence lengths
            num_iterations: number of decoding iterations
            confidence_threshold: accept predictions above this
            
        Returns:
            predicted_strings: (batch, seq_len) — predicted string indices (0-5)
            confidences: (batch, seq_len) — prediction confidence scores
        """
        self.eval()
        B, L = pitches.shape
        device = pitches.device
        
        # Create padding mask
        if lengths is not None:
            padding_mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = None
        
        with torch.no_grad():
            logits = self.forward(pitches, time_shifts, durations, padding_mask)
            probs = F.softmax(logits, dim=-1)
            
            # Get predictions and confidences
            confidences, predicted = probs.max(dim=-1)
            
            # Apply validity constraints: check if (string, fret) is physically possible
            predicted = self._apply_constraints(predicted, pitches, confidences)
        
        return predicted, confidences
    
    def _apply_constraints(self, predicted, pitches, confidences):
        """Apply physical guitar constraints to predictions.
        
        Ensure predicted (string, fret) combination is physically playable.
        If not, fall back to the next most likely valid string.
        """
        # Standard tuning open string pitches (string 0-5)
        open_pitches = torch.tensor([64, 59, 55, 50, 45, 40], 
                                     device=predicted.device)
        max_fret = 19
        
        B, L = predicted.shape
        for b in range(B):
            for i in range(L):
                string = predicted[b, i].item()
                pitch = pitches[b, i].item()
                if pitch == 0:  # padding
                    continue
                
                fret = pitch - open_pitches[string].item()
                if fret < 0 or fret > max_fret:
                    # Invalid: find best valid string
                    for s in range(6):
                        f = pitch - open_pitches[s].item()
                        if 0 <= f <= max_fret:
                            predicted[b, i] = s
                            break
        
        return predicted
    
    def get_string_probabilities(self, pitches, time_shifts, durations,
                                  lengths=None):
        """Get per-note string probability distributions.
        
        Used for integration with Viterbi DP in string_assigner.py.
        
        Returns:
            probs: (batch, seq_len, 6) — probability for each string
        """
        self.eval()
        B, L = pitches.shape
        device = pitches.device
        
        if lengths is not None:
            padding_mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = None
        
        with torch.no_grad():
            logits = self.forward(pitches, time_shifts, durations, padding_mask)
            probs = F.softmax(logits, dim=-1)
        
        return probs


def create_mlm_mask(strings, mask_ratio=0.15, ignore_index=-1):
    """Create masked language modeling mask for training.
    
    Args:
        strings: (batch, seq_len) — ground truth string labels
        mask_ratio: fraction of positions to mask
        ignore_index: padding value to ignore
        
    Returns:
        masked_strings: (batch, seq_len) — strings with some positions masked
        mask: (batch, seq_len) — True at positions that were masked
    """
    B, L = strings.shape
    
    # Don't mask padded positions
    valid = strings != ignore_index
    
    # Random mask
    rand = torch.rand(B, L, device=strings.device)
    mask = (rand < mask_ratio) & valid
    
    # Ensure at least one position is masked per sequence
    for b in range(B):
        if not mask[b].any() and valid[b].any():
            valid_indices = valid[b].nonzero(as_tuple=True)[0]
            idx = valid_indices[torch.randint(len(valid_indices), (1,))]
            mask[b, idx] = True
    
    return mask
