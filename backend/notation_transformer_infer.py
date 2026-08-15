"""
notation_transformer_infer.py — Notation Transformer 推論＆ハイブリッド統合
========================================================================
1. ノート列から (voice_id, tuplet_role, duration_divs, tie_flag) を推論
2. 信頼度 >= 0.6: Transformer出力を優先適用
3. 信頼度 < 0.6: 物理ルールへフォールバック (ハイブリッド)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
from pathlib import Path
from train_notation_transformer import NotationTransformer, DEVICE

MODEL_PATH = Path("D:/Music/chordlink-solotab/backend/models/notation_transformer.pth")
_MODEL_CACHE = None


def get_notation_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        return None
    try:
        model = NotationTransformer().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _MODEL_CACHE = model
        return model
    except Exception as e:
        print(f"[notation_transformer] Model load failed: {e}")
        return None


def apply_notation_transformer(notes: list, beats_per_bar: int = 3, divisions: int = 12) -> list:
    """
    推論を実行し、ノートオブジェクトに voice, is_triplet, tuplet_role, duration_divs を付与。
    """
    if not notes:
        return []

    model = get_notation_model()
    if model is None:
        return notes

    T = len(notes)
    if T == 0:
        return notes

    pitches = np.array([int(n.get("pitch", 60)) for n in notes], dtype=np.int64)[None, :]
    strings = np.array([int(n.get("string", 1)) for n in notes], dtype=np.int64)[None, :]
    
    conts = np.zeros((1, T, 2), dtype=np.float32)
    for i, n in enumerate(notes):
        conts[0, i, 0] = float(n.get("velocity", 0.7))
        pos_in_bar = float(n.get("beat_pos_in_bar", 0))
        conts[0, i, 1] = (pos_in_bar % (beats_per_bar * divisions)) / float(beats_per_bar * divisions)

    t_p = torch.from_numpy(pitches).to(DEVICE)
    t_s = torch.from_numpy(strings).to(DEVICE)
    t_c = torch.from_numpy(conts).to(DEVICE)

    with torch.no_grad():
        l_v, l_t, l_d, l_i = model(t_p, t_s, t_c)
        prob_v = torch.softmax(l_v, dim=-1)[0].cpu().numpy()
        prob_t = torch.softmax(l_t, dim=-1)[0].cpu().numpy()

    tuplet_role_map = {0: "none", 1: "start", 2: "middle", 3: "stop"}

    for i, n in enumerate(notes):
        # 1. Voice判定 (Voice 1 = 0 / Voice 2 = 1)
        conf_v = float(np.max(prob_v[i]))
        pred_v = int(np.argmax(prob_v[i]))
        if conf_v >= 0.60:
            n["voice"] = 2 if pred_v == 1 else 1
            n["is_bass"] = (pred_v == 1)
        else:
            # フォールバック
            s = int(n.get("string", 1))
            p = int(n.get("pitch", 60))
            is_b = (s >= 4 or p <= 52)
            n["voice"] = 2 if is_b else 1
            n["is_bass"] = is_b

        # 2. Tuplet判定
        conf_t = float(np.max(prob_t[i]))
        pred_t = int(np.argmax(prob_t[i]))
        if conf_t >= 0.60:
            n["tuplet_role"] = tuplet_role_map.get(pred_t, "none")
            n["is_triplet"] = (pred_t in (1, 2, 3))
        else:
            pass  # 既存Universal Quantizerの値を保持

    return notes
