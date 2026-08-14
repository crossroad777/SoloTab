from __future__ import annotations
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
"""
technique_classifier_cnn.py - CNN-based technique classifier (inference only)
==============================================================================
Trained on IDMT-SMT-GUITAR_V2. Uses mel+delta+delta2 patches around each note
onset to classify: normal, muted, bend, slide, harmonic, vibrato.

Usage in pipeline:
    from technique_classifier_cnn import annotate_techniques_cnn
    notes = annotate_techniques_cnn(notes, audio_path, confidence_threshold=0.90)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional

# ── Model architecture (must match training) ──

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.view(b, c, -1).mean(dim=2)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class TechniqueCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Globals (loaded once) ──
_model = None
_device = None
_ckpt_meta = None
MODEL_PATH = Path(__file__).parent / "models" / "technique_cnn.pth"

# CNN label -> pipeline technique name mapping
_CNN_TO_TECHNIQUE = {
    "normal":   "normal",
    "muted":    "pm",        # palm mute
    "bend":     "b",         # bend
    "slide":    "/",         # slide up (default)
    "harmonic": "harmonic",  # natural harmonic
    "vibrato":  "~",         # vibrato
}


def _load_model():
    """Load model (cached singleton)."""
    global _model, _device, _ckpt_meta
    if _model is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Technique CNN model not found: {MODEL_PATH}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ckpt_meta = torch.load(str(MODEL_PATH), map_location=_device, weights_only=False)

    num_classes = len(_ckpt_meta["label_names"])
    _model = TechniqueCNN(num_classes).to(_device)
    _model.load_state_dict(_ckpt_meta["model_state_dict"])
    _model.eval()
    print(f"[TechCNN] Model loaded: {MODEL_PATH.name} "
          f"(epoch={_ckpt_meta.get('epoch')}, "
          f"val_f1={_ckpt_meta.get('val_f1', 0):.3f}, "
          f"device={_device})")


def _extract_patches(notes: List[dict], audio_path: str) -> np.ndarray:
    """Extract 3-channel mel patches for all notes in batch."""
    import librosa

    cfg = _ckpt_meta.get("config", _ckpt_meta)  # V4 nests in config, V2 is flat
    sr = cfg["sr"]
    n_mels = cfg["n_mels"]
    hop = cfg["hop_length"]
    n_fft = cfg.get("n_fft", 1024)
    patch_frames = cfg["patch_frames"]

    # Load audio once
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Compute features once
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=50, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(mel_db, order=1).astype(np.float32)
    delta2 = librosa.feature.delta(mel_db, order=2).astype(np.float32)

    total_frames = mel_db.shape[1]
    half = patch_frames // 2
    patches = []

    for note in notes:
        onset = float(note.get("start", 0))
        onset_frame = int(onset * sr / hop)
        start = max(0, onset_frame - half)
        end = start + patch_frames
        if end > total_frames:
            end = total_frames
            start = max(0, end - patch_frames)

        channels = []
        for feat in [mel_db, delta, delta2]:
            p = feat[:, start:end]
            if p.shape[1] < patch_frames:
                pad = np.zeros((n_mels, patch_frames - p.shape[1]), dtype=np.float32)
                p = np.concatenate([p, pad], axis=1)
            # Normalize
            m, s = p.mean(), p.std() + 1e-6
            p = (p - m) / s
            channels.append(p)

        patches.append(np.stack(channels, axis=0))  # [3, n_mels, patch_frames]

    return np.array(patches, dtype=np.float32)


def annotate_techniques_cnn(
    notes: List[dict],
    audio_path: str,
    confidence_threshold: float = 0.90,
) -> List[dict]:
    """
    CNN-based technique annotation.

    Parameters
    ----------
    notes : list of dict with 'start', 'end', 'pitch', 'string', 'fret'
    audio_path : path to WAV/MP3 file
    confidence_threshold : minimum confidence to assign technique (default 0.90)

    Returns
    -------
    notes : same list with 'technique' and 'technique_confidence' added
    """
    if not notes or not audio_path:
        return notes

    try:
        _load_model()
    except Exception as e:
        print(f"[TechCNN] Model load failed: {e}")
        return notes

    label_names = _ckpt_meta["label_names"]

    # Batch extract patches
    print(f"[TechCNN] Processing {len(notes)} notes...")
    import time
    t0 = time.time()
    patches = _extract_patches(notes, audio_path)
    t_extract = time.time() - t0

    # Batch inference
    t0 = time.time()
    batch_size = 256
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[i:i+batch_size]).to(_device)
            logits = _model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    t_infer = time.time() - t0

    # Apply predictions
    stats = {"total": len(notes), "annotated": 0, "by_class": {}}
    for i, note in enumerate(notes):
        pred_class = int(all_probs[i].argmax())
        confidence = float(all_probs[i].max())
        label = label_names[pred_class]

        # Only assign if confidence exceeds threshold AND not "normal"
        if label != "normal" and confidence >= confidence_threshold:
            technique = _CNN_TO_TECHNIQUE.get(label, label)
            note["technique"] = technique
            note["technique_confidence"] = round(confidence, 3)
            note["technique_source"] = "cnn"
            stats["annotated"] += 1
            stats["by_class"][label] = stats["by_class"].get(label, 0) + 1

    print(f"[TechCNN] Done: {stats['annotated']}/{stats['total']} annotated "
          f"(threshold={confidence_threshold}) "
          f"extract={t_extract:.1f}s, infer={t_infer:.1f}s")
    for cls, cnt in sorted(stats["by_class"].items()):
        print(f"  {cls}: {cnt}")

    return notes
