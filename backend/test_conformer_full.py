import sys
import os
import torch

project_root = r"D:\Music\nextchord-solotab\music-transcription\python"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from model.architecture import GuitarTabCRNN

def test():
    print("Testing Future Architecture - Multi-channel CNN + Conformer (No batch_first GRU)")
    
    # [Batch, Channels(RGB), Freq(CQT Bins), Time(Frames)]
    batch_size = 2
    time_frames = 100
    x = torch.randn(batch_size, 3, 168, time_frames)
    
    model = GuitarTabCRNN(
        num_frames_rnn_input_dim=1280, 
        rnn_type="CONFORMER",
        cnn_input_channels=3
    )
    
    print("Model 'GuitarTabCRNN' instantiated successfully with Conformer.")
    
    onset_logits, fret_logits = model(x)
    print("Forward pass completed successfully without crash!")
    
    assert onset_logits.shape == (batch_size, time_frames, 6), f"Wrong onset shape: {onset_logits.shape}"
    assert fret_logits.shape == (batch_size, time_frames, 6, 22), f"Wrong fret shape: {fret_logits.shape}"
    
    print("Dimension Assertion PASSED:")
    print(f" -> Onset Logits : {onset_logits.shape}")
    print(f" -> Fret Logits  : {fret_logits.shape}")

if __name__ == "__main__":
    test()
