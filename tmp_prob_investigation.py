import sys
import numpy as np
import torch
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")
from guitar_transcriber import _load_model, _extract_cqt, SAMPLE_RATE, HOP_LENGTH
import librosa

wav_path = r"D:\Music\nextchord-solotab\uploads\forbidden_games_test.wav"
model, device = _load_model()
cqt_features = _extract_cqt(wav_path)
channels, n_bins, total_frames = cqt_features.shape

with torch.no_grad():
    x = cqt_features[:, :, :1000].unsqueeze(0).to(device)
    onset_logits, fret_logits = model(x)
    onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
    fret_preds = torch.argmax(fret_logits[0], dim=-1).cpu().numpy()

# Look at frames around t = 1.90s
target_frame = int(1.90 * SAMPLE_RATE / HOP_LENGTH)
for f in range(target_frame - 2, target_frame + 5):
    t_sec = f * HOP_LENGTH / SAMPLE_RATE
    probs = onset_probs[f]
    frets = fret_preds[f]
    print(f"Time {t_sec:.3f}s (Frame {f}):")
    print(f"  Probs: {np.round(probs, 3)}")
    print(f"  Frets: {frets}")
