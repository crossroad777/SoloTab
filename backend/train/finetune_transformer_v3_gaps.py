"""
backend/train/finetune_transformer_v3_gaps.py
=============================================
TASK-903: GAPSデータセットによる「開放弦・ローポジション美学」の再学習
"""

import os
import sys
import glob
import math
import random
import pathlib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("backend"))

from fingering_model_v3 import FingeringTransformer
from solotab_utils import STANDARD_TUNING

# 開放弦ピッチ
OPEN_PITCHES = {40: 6, 45: 5, 50: 4, 55: 3, 59: 2, 64: 1} # pitch -> string


class GAPSClassicalDataset(Dataset):
    def __init__(self, sample_limit: int = 1500, ctx_len: int = 16):
        self.ctx_len = ctx_len
        self.samples = []
        
        # 1. Ground Truth コーパス (Romance, Lagrima, GAPS等) から高品質ソロギターノート列を収集
        gt_json_path = pathlib.Path("backend/ground_truth/romance_forbidden_games.json")
        if gt_json_path.exists():
            with open(gt_json_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            seq = []
            for m in gt_data["measures_detailed"]:
                for n in m["notes"]:
                    seq.append({
                        "pitch": n["pitch"],
                        "string": n["string"],
                        "fret": n["fret"],
                        "duration": 0.22
                    })
            if len(seq) > ctx_len:
                self._extract_windows(seq)

        # 2. GProTab / GAPS 配下のクラシック・アコースティックデータ
        gaps_candidates = glob.glob("datasets/gprotab_downloads/anonymous/*.gp*") + \
                          glob.glob("datasets/gprotab_downloads/anonimo-romance/*.gp*") + \
                          glob.glob("datasets/gaps/gaps_v1/midi/*fine-aligned.mid")[:sample_limit]
                          
        import mido
        for mid_path in gaps_candidates[:sample_limit]:
            try:
                if mid_path.endswith(".mid"):
                    mid = mido.MidiFile(mid_path)
                    notes_seq = []
                    for t in mid.tracks:
                        for msg in t:
                            if msg.type == 'note_on' and msg.velocity > 0:
                                p = msg.note
                                # 最適なソロギター開放弦/ローポジションを付与
                                s = OPEN_PITCHES.get(p, 1 if p >= 64 else (2 if p >= 59 else (3 if p >= 55 else (4 if p >= 50 else (5 if p >= 45 else 6)))))
                                f = max(0, min(24, p - STANDARD_TUNING[6 - s]))
                                notes_seq.append({"pitch": p, "string": s, "fret": f, "duration": 0.25})
                    if len(notes_seq) > ctx_len:
                        self._extract_windows(notes_seq)
            except Exception:
                continue

    def _extract_windows(self, seq):
        for i in range(1, len(seq)):
            start_idx = max(0, i - self.ctx_len)
            ctx = seq[start_idx:i]
            target = seq[i]
            
            pitches, strings, frets, durations, intervals = [], [], [], [], []
            prev_p = 0
            for cn in ctx:
                p = min(127, max(0, cn["pitch"]))
                s = min(6, max(0, cn["string"]))
                f = min(24, max(0, cn["fret"]))
                dur = min(31, max(0, int(cn["duration"] * 8)))
                interval = min(48, max(0, (p - prev_p) + 24)) if prev_p else 24
                pitches.append(p)
                strings.append(s)
                frets.append(f)
                durations.append(dur)
                intervals.append(interval)
                prev_p = p
                
            pad_len = self.ctx_len - len(pitches)
            pitches = [0] * pad_len + pitches
            strings = [0] * pad_len + strings
            frets = [0] * pad_len + frets
            durations = [0] * pad_len + durations
            intervals = [24] * pad_len + intervals
            
            tp = min(127, max(0, target["pitch"]))
            td = min(31, max(0, int(target["duration"] * 8)))
            ti = min(48, max(0, (tp - prev_p) + 24)) if prev_p else 24
            
            recent_frets = [f for f in frets[-8:] if f > 0]
            pc = min(24, int(np.median(recent_frets))) if recent_frets else 0
            
            target_str = target["string"] - 1 # 0-5
            self.samples.append({
                "ctx_p": torch.tensor(pitches, dtype=torch.long),
                "ctx_s": torch.tensor(strings, dtype=torch.long),
                "ctx_f": torch.tensor(frets, dtype=torch.long),
                "ctx_d": torch.tensor(durations, dtype=torch.long),
                "ctx_i": torch.tensor(intervals, dtype=torch.long),
                "t_p": torch.tensor(tp, dtype=torch.long),
                "t_d": torch.tensor(td, dtype=torch.long),
                "t_i": torch.tensor(ti, dtype=torch.long),
                "p_c": torch.tensor(pc, dtype=torch.long),
                "target_str": torch.tensor(target_str, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AestheticFingeringLoss(nn.Module):
    """
    法則2 (開放弦選好) + 法則4 (ポジション固執・過度な跳躍ペナルティ) を適用した損失関数
    """
    def __init__(self, lambda_open: float = 0.40, lambda_jump: float = 0.50):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_open = lambda_open
        self.lambda_jump = lambda_jump

    def forward(self, logits, targets, target_pitches, pos_contexts):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        
        open_loss = torch.tensor(0.0, device=logits.device)
        jump_loss = torch.tensor(0.0, device=logits.device)
        
        tuning = STANDARD_TUNING
        
        for b in range(logits.size(0)):
            p = int(target_pitches[b].item())
            pc = int(pos_contexts[b].item())
            
            # 法則2: 開放弦選好 (Open String Aesthetic)
            if p in OPEN_PITCHES:
                open_str_idx = OPEN_PITCHES[p] - 1 # 0-5
                open_loss -= torch.log(probs[b, open_str_idx] + 1e-7)
                
            # 法則4: ポジション固執と過度なハイポジション跳躍ペナルティ
            for s_idx in range(6):
                sn = s_idx + 1
                f = p - tuning[6 - sn]
                if 0 <= f <= 24:
                    if f > 0 and pc > 0 and abs(f - pc) > 4:
                        jump_loss += probs[b, s_idx] * float(abs(f - pc))
                        
        total_loss = ce_loss + self.lambda_open * (open_loss / max(1, logits.size(0))) + self.lambda_jump * (jump_loss / max(1, logits.size(0)))
        return total_loss, ce_loss.item()


def finetune_v3():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[finetune_v3] Using device: {device}")
    
    # 既存の学習済みモデルをロード
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'gp_training_data', 'v3', 'models')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'fingering_transformer_v3_best.pt')
    finetuned_model_path = os.path.join(model_dir, 'fingering_transformer_v3_finetuned.pt')
    
    model = FingeringTransformer()
    if os.path.exists(finetuned_model_path):
        state = torch.load(finetuned_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        print(f"[finetune_v3] Loaded existing finetuned weights from: {finetuned_model_path}")
    elif os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        print(f"[finetune_v3] Loaded base weights from: {best_model_path}")
    else:
        print("[finetune_v3] Base weights not found, initializing fresh model")

    model.to(device)
    model.train()
    
    dataset = GAPSClassicalDataset(sample_limit=2000)
    print(f"[finetune_v3] Prepared {len(dataset)} training windows from GAPS/Classical corpus")
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = AestheticFingeringLoss(lambda_open=0.40, lambda_jump=0.50)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 5
    epoch_logs = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in loader:
            optimizer.zero_grad()
            ctx_p = batch["ctx_p"].to(device)
            ctx_s = batch["ctx_s"].to(device)
            ctx_f = batch["ctx_f"].to(device)
            ctx_d = batch["ctx_d"].to(device)
            ctx_i = batch["ctx_i"].to(device)
            t_p = batch["t_p"].to(device)
            t_d = batch["t_d"].to(device)
            t_i = batch["t_i"].to(device)
            p_c = batch["p_c"].to(device)
            targets = batch["target_str"].to(device)
            
            logits = model(ctx_p, ctx_s, ctx_f, ctx_d, ctx_i, t_p, t_d, t_i, p_c)
            loss, ce_val = criterion(logits, targets, t_p, p_c)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        acc = round(correct / max(1, total), 4)
        avg_loss = round(total_loss / max(1, len(loader)), 4)
        epoch_logs.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "training_accuracy": acc
        })
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {acc:.2%}")
        
    # 保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': epoch_logs[-1]["training_accuracy"],
        'epoch': epochs
    }, finetuned_model_path)
    print(f"[finetune_v3] Saved finetuned model to: {finetuned_model_path}")
    
    return epoch_logs, finetuned_model_path

if __name__ == "__main__":
    finetune_v3()
