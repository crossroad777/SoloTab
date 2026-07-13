"""
fingering_transformer_v1.py — V1 FingeringTransformer (reconstructed from weights)

Architecture (from state_dict):
  input_proj: Linear(6, 64)   — 6 input features per token
  pos_enc: (1, 1024, 64)      — sinusoidal positional encoding  
  encoder: TransformerEncoder(4 layers, d_model=64, nhead=?, ffn=256)
  fc: Linear(64, 5)           — 5-class output (strings 1-5? or 2-6?)

Input: sequence of 6-feature tokens (pitch_norm, string, fret, duration, interval, position_context)
Output: 5-class string prediction
"""
import math
import torch
import torch.nn as nn


class FingeringTransformerV1(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, 
                 dim_feedforward=256, n_input=6, n_output=5, max_len=1024):
        super().__init__()
        self.d_model = d_model
        
        # Input projection: 6 features -> d_model
        self.input_proj = nn.Linear(n_input, d_model)
        
        # Positional encoding (stored as buffer)
        self.register_buffer('pos_enc', torch.zeros(1, max_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Linear(d_model, n_output)
    
    def forward(self, x):
        """
        x: (batch, seq_len, 6) — 6 features per token
        returns: (batch, 5) — string class logits
        """
        # Project input
        x = self.input_proj(x)  # (B, L, 64)
        
        # Add positional encoding
        x = x + self.pos_enc[:, :x.size(1), :]
        
        # Transformer encoder
        x = self.encoder(x)  # (B, L, 64)
        
        # Use last token output
        x = x[:, -1, :]  # (B, 64)
        
        # Classification
        logits = self.fc(x)  # (B, 5)
        return logits


def load_v1_model(model_path=None, device='cpu'):
    """Load the V1 FingeringTransformer with pretrained weights."""
    import os
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'finger_transformer_v1.pth')
    
    # Try nhead=4 first (64/4=16), if fails try nhead=2
    for nhead in [4, 2, 1]:
        try:
            model = FingeringTransformerV1(nhead=nhead)
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            return model
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not load V1 model from {model_path}")
