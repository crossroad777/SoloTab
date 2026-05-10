"""
technique_classifier_cnn.py — CNNベースのテクニック分類
=======================================================
ノートにtechnique属性 (normal / palm_mute / harmonic) を付与。

旧 ensemble_transcriber.py L509-582 + technique_classifier.py L45-115 から統合。
"""

import numpy as np
import librosa
from typing import List, Dict, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Technique classes (technique_classifier.py L45-53 より) ---
TECHNIQUE_CLASSES = [
    'normal',        # 0: Standard picking
    'palm_mute',     # 1: Palm mute
    'harmonic',      # 2: Natural/artificial harmonics
    'bend',          # 3: Bend
    'slide',         # 4: Slide
    'vibrato',       # 5: Vibrato
    'dead_note',     # 6: Dead/ghost note
]
NUM_CLASSES = len(TECHNIQUE_CLASSES)


# --- CNN Model (technique_classifier.py L77-115 より) ---
class TechniqueClassifierCNN(nn.Module):
    """
    Small CNN for guitar technique classification from Mel spectrograms.
    Input: [B, 1, N_MELS, time_frames]
    Output: [B, NUM_CLASSES]
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --- Inference ---
def annotate_techniques(wav_path: str, notes: List[Dict],
                        report: Optional[Callable] = None) -> List[Dict]:
    """
    テクニック分類器でノートにtechnique属性を付与。
    technique: normal / palm_mute / harmonic / bend / slide / vibrato / dead_note
    """
    model_path = Path(__file__).parent.parent / "generated" / "technique_classifier" / "best_model.pt"
    if not model_path.exists():
        if report:
            report("テクニック分類: モデル未検出、スキップ")
        return notes

    try:
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        # チェックポイントにクラス数が含まれている場合はそれを使用
        n_classes = len(checkpoint.get('classes', TECHNIQUE_CLASSES))
        classes = checkpoint.get('classes', TECHNIQUE_CLASSES)
        model = TechniqueClassifierCNN(n_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load audio once
        y, sr = librosa.load(wav_path, sr=22050, mono=True)

        tech_counts = {}
        seg_dur = 0.3
        seg_samples = int(seg_dur * 22050)

        for note in notes:
            onset = note.get('start', 0)
            start_sample = max(0, int(onset * 22050) - int(0.02 * 22050))
            end_sample = start_sample + seg_samples

            if end_sample > len(y):
                note['technique'] = 'normal'
                continue

            segment = y[start_sample:end_sample]
            if len(segment) < seg_samples:
                segment = np.pad(segment, (0, seg_samples - len(segment)))

            mel = librosa.feature.melspectrogram(
                y=segment.astype(np.float32), sr=22050,
                n_mels=64, n_fft=1024, hop_length=256
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

            x = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]
                pred = probs.argmax().item()
                confidence = probs[pred].item()

            tech_name = classes[pred] if pred < len(classes) else 'normal'
            # 信頼度閾値: normal以外は高確信(>0.95)の場合のみ採用
            if tech_name != 'normal' and confidence < 0.95:
                tech_name = 'normal'
            note['technique'] = tech_name
            note['technique_confidence'] = round(confidence, 3)
            tech_counts[tech_name] = tech_counts.get(tech_name, 0) + 1

        tech_str = ', '.join(f'{k}:{v}' for k, v in sorted(tech_counts.items()))
        if report:
            report(f"テクニック推定完了: {tech_str}")

    except Exception as e:
        if report:
            report(f"テクニック推定エラー: {e}")
        for note in notes:
            note['technique'] = 'normal'

    return notes
