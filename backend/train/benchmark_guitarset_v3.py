"""V3 Transformer で GuitarSet突合 + Attention解析"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from fingering_model_v3 import FingeringTransformer

JAMS_DIR = Path(r"D:\Music\Datasets\GuitarSet\annotation")
MODEL_PATH = Path(__file__).parent.parent.parent / "gp_training_data" / "v3" / "models" / "fingering_transformer_v3_best.pt"
OPEN_PITCH = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
args = ckpt["args"]
model = FingeringTransformer(
    d_model=args["d_model"], nhead=args["nhead"],
    num_layers=args["num_layers"], embed_dim=args["embed_dim"],
    dropout=0.0,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, epoch {ckpt['epoch']}")

jams_files = sorted(JAMS_DIR.glob("*.jams"))
print(f"JAMS: {len(jams_files)} files")

all_correct = 0; all_total = 0
amb_correct = 0; amb_total = 0
tol1_correct = 0
string_correct = Counter(); string_total = Counter()
cand_correct = Counter(); cand_total = Counter()
confusion = defaultdict(Counter)

for jf in jams_files:
    try:
        data = json.load(open(jf, encoding="utf-8"))
    except:
        continue
    
    notes = []
    for ann in data["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        ds = int(ann["annotation_metadata"]["data_source"])
        s = 6 - ds
        for entry in ann["data"]:
            notes.append((entry["time"], round(entry["value"]), s, entry.get("duration", 0.5)))
    
    notes.sort()
    if len(notes) < 20:
        continue
    
    ctx_len = 16
    batches = {"cp":[], "cs":[], "cf":[], "cd":[], "ci":[], "tp":[], "td":[], "ti":[], "pc":[], "gt":[], "amb":[], "nc":[]}
    
    for i in range(ctx_len, len(notes)):
        ctx = notes[i-ctx_len:i]
        target = notes[i]
        
        cp = [n[1] for n in ctx]
        cs = [n[2] for n in ctx]
        cf = [max(0, min(24, n[1] - OPEN_PITCH.get(n[2], 40))) for n in ctx]
        cd = [min(31, int(n[3] * 10)) for n in ctx]
        ci = [0] + [max(-24, min(24, ctx[j][1] - ctx[j-1][1])) + 24 for j in range(1, len(ctx))]
        
        recent_frets = [f for f in cf[-8:] if f > 0]
        pc = int(sum(recent_frets) / len(recent_frets)) if recent_frets else 0
        
        nc = sum(1 for s in range(1, 7) if 0 <= target[1] - OPEN_PITCH[s] <= 24)
        td = min(31, int(target[3] * 10))
        ti_val = max(-24, min(24, target[1] - ctx[-1][1])) + 24
        
        batches["cp"].append(cp); batches["cs"].append(cs); batches["cf"].append(cf)
        batches["cd"].append(cd); batches["ci"].append(ci)
        batches["tp"].append(target[1]); batches["td"].append(td); batches["ti"].append(ti_val)
        batches["pc"].append(min(24, pc)); batches["gt"].append(target[2])
        batches["amb"].append(nc > 1); batches["nc"].append(nc)
    
    if not batches["cp"]:
        continue
    
    with torch.no_grad():
        logits = model(
            torch.tensor(batches["cp"]).to(device),
            torch.tensor(batches["cs"]).to(device),
            torch.tensor(batches["cf"]).to(device),
            torch.tensor(batches["cd"]).to(device),
            torch.tensor(batches["ci"]).to(device),
            torch.tensor(batches["tp"]).to(device),
            torch.tensor(batches["td"]).to(device),
            torch.tensor(batches["ti"]).to(device),
            torch.tensor(batches["pc"]).to(device),
        )
        preds = (logits.argmax(dim=-1) + 1).cpu().tolist()
    
    for pred, gt, amb, nc in zip(preds, batches["gt"], batches["amb"], batches["nc"]):
        all_total += 1
        hit = (pred == gt)
        if hit: all_correct += 1
        if abs(pred - gt) <= 1: tol1_correct += 1
        if amb:
            amb_total += 1
            if hit: amb_correct += 1
        string_total[gt] += 1
        if hit: string_correct[gt] += 1
        cand_total[nc] += 1
        if hit: cand_correct[nc] += 1
        confusion[gt][pred] += 1

print(f"\n{'='*60}")
print(f"Overall:    {all_correct/all_total:.4f} ({all_correct:,}/{all_total:,})")
print(f"Ambiguous:  {amb_correct/amb_total:.4f} ({amb_correct:,}/{amb_total:,})")
print(f"Tol +/-1:   {tol1_correct/all_total:.4f} ({tol1_correct:,}/{all_total:,})")

print(f"\n--- By String ---")
for s in range(1, 7):
    if string_total[s]:
        print(f"  Str{s}: {string_correct[s]/string_total[s]:.4f} ({string_correct[s]:,}/{string_total[s]:,})")

print(f"\n--- By Candidate Count ---")
for nc in sorted(cand_total.keys()):
    if cand_total[nc]:
        print(f"  {nc} cand: {cand_correct[nc]/cand_total[nc]:.4f} ({cand_correct[nc]:,}/{cand_total[nc]:,})")

print(f"\n--- Confusion Matrix ---")
print(f"       ", end="")
for p in range(1, 7): print(f" Pred{p}", end="")
print()
for gt in range(1, 7):
    print(f"  GT{gt}:", end="")
    row_total = sum(confusion[gt].values())
    for p in range(1, 7):
        c = confusion[gt][p]
        print(f" {c:5d}" if c else "     .", end="")
    if row_total:
        print(f"  | {confusion[gt][gt]/row_total*100:.1f}%")
