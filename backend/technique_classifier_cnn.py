"""
technique_classifier_cnn.py — CNNベースのテクニック分類
=======================================================
ノートにtechnique属性 (normal / palm_mute / harmonic) を付与。

旧 ensemble_transcriber.py L509-582 から抽出。
"""

import numpy as np
import librosa
from typing import List, Dict, Optional, Callable
from pathlib import Path


def annotate_techniques(wav_path: str, notes: List[Dict],
                        report: Optional[Callable] = None) -> List[Dict]:
    """
    テクニック分類器でノートにtechnique属性を付与。
    technique: normal / palm_mute / harmonic
    """
    model_path = Path(__file__).parent.parent / "generated" / "technique_classifier" / "best_model.pt"
    if not model_path.exists():
        if report:
            report("テクニック分類: モデル未検出、スキップ")
        return notes

    try:
        import torch
        from string_classifier import TechniqueClassifierCNN, TECHNIQUE_CLASSES

        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model = TechniqueClassifierCNN()
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

            tech_name = TECHNIQUE_CLASSES[pred] if pred < len(TECHNIQUE_CLASSES) else 'normal'
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
