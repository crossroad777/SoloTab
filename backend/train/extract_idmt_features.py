"""
IDMT-SMT-GUITAR_V2 → CNN学習データ統合
======================================
IDMT単音データ(弦+ピッチ+音声)をGuitarSet形式のfeatureに変換し、
CNN学習データとして統合。これでギター/プレイヤーの多様性を向上。
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import xml.etree.ElementTree as ET
import numpy as np

from string_classifier import compute_cqt, SR, HOP_LENGTH, CONTEXT_FRAMES, N_BINS

IDMT_BASE = r"D:\Music\Datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2"


def extract_idmt_features():
    """IDMTデータからCNN学習用featureを抽出"""
    audio_files = glob.glob(os.path.join(IDMT_BASE, "**", "audio", "*.wav"), recursive=True)
    # 単音のみ（コードは除く）
    audio_files = [f for f in audio_files if "Chord" not in os.path.dirname(f)]
    print(f"IDMT single-note audio files: {len(audio_files)}")

    features = []
    skipped = 0

    for af in sorted(audio_files):
        # 対応するXMLアノテーション
        ann_dir = os.path.join(os.path.dirname(os.path.dirname(af)), "annotation")
        xml_path = os.path.join(ann_dir, os.path.basename(af).replace(".wav", ".xml"))

        if not os.path.exists(xml_path):
            skipped += 1
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except:
            skipped += 1
            continue

        # CQT計算
        try:
            cqt = compute_cqt(af)
        except:
            skipped += 1
            continue

        # XMLからイベントを抽出
        for event in root.findall(".//event"):
            pitch_elem = event.find("pitch")
            string_elem = event.find("stringNumber")
            fret_elem = event.find("fretNumber")
            onset_elem = event.find("onsetSec")

            if any(e is None for e in [pitch_elem, string_elem, onset_elem]):
                continue

            pitch = int(pitch_elem.text)
            idmt_string = int(string_elem.text)  # IDMT: 1=6弦(E2), 6=1弦(E4)
            onset = float(onset_elem.text)

            # IDMT → GuitarSet弦番号: IDMT 1→GS 6, IDMT 6→GS 1
            gs_string = 7 - idmt_string

            if gs_string < 1 or gs_string > 6:
                continue
            if pitch < 30 or pitch > 96:
                continue

            frame_idx = int(onset * SR / HOP_LENGTH)
            half_ctx = CONTEXT_FRAMES // 2
            if frame_idx - half_ctx < 0 or frame_idx + half_ctx >= cqt.shape[1]:
                continue

            patch = cqt[:, frame_idx - half_ctx:frame_idx + half_ctx + 1]

            features.append({
                "patch": patch.astype(np.float32),
                "pitch": pitch,
                "string": gs_string,
            })

    print(f"Extracted: {len(features)} features, Skipped: {skipped}")
    return features


if __name__ == "__main__":
    features = extract_idmt_features()
    from collections import Counter
    string_dist = Counter(f["string"] for f in features)
    print("\nString distribution:")
    for s in range(1, 7):
        print(f"  S{s}: {string_dist.get(s, 0)}")
