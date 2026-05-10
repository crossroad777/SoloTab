"""
analyze_fingering_rules.py — 人間の運指法則を定量化
===================================================
1. Attention重み解析: どの文脈位置が弦選択に影響するか
2. 特徴量Ablation: duration/interval/position各特徴量の寄与度
3. 統計的ルール抽出: ピッチ間隔と弦変更の関係
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fingering_model_v3 import FingeringTransformer

MODEL_PATH = Path(__file__).parent.parent.parent / "gp_training_data" / "v3" / "models" / "fingering_transformer_v3_best.pt"
DATA_DIR = Path(__file__).parent.parent.parent / "gp_training_data" / "v3"
JAMS_DIR = Path(r"D:\Music\Datasets\GuitarSet\annotation")
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

# ========================================
# Part 1: Attention Weight Analysis
# ========================================
print("=" * 60)
print("Part 1: Attention Weight Analysis")
print("=" * 60)

# Hook to capture attention weights
attn_weights_all = []

def capture_attention(module, input, output):
    # TransformerEncoderLayer stores attn weights if we use need_weights
    pass

# Modify to get attention weights by running manually
# Use a subset of test data
test_samples = []
with open(DATA_DIR / "fingering_test.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        test_samples.append(json.loads(line))

print(f"  Test samples: {len(test_samples)}")

# Batch inference with attention extraction
batch_size = 2048
position_importance = np.zeros(16)  # context length
n_batches = 0

for start in range(0, len(test_samples), batch_size):
    batch = test_samples[start:start+batch_size]
    
    ctx_p = torch.tensor([s["context_pitches"] for s in batch]).to(device)
    ctx_s = torch.tensor([s["context_strings"] for s in batch]).to(device)
    ctx_f = torch.tensor([[min(f,24) for f in s["context_frets"]] for s in batch]).to(device)
    ctx_d = torch.tensor([s["context_durations"] for s in batch]).to(device)
    ctx_i = torch.tensor([s["context_intervals"] for s in batch]).to(device)
    tgt_p = torch.tensor([s["target_pitch"] for s in batch]).to(device)
    tgt_d = torch.tensor([s["target_duration"] for s in batch]).to(device)
    tgt_i = torch.tensor([s["target_interval"] for s in batch]).to(device)
    pos_c = torch.tensor([s["position_context"] for s in batch]).to(device)
    
    with torch.no_grad():
        # Get intermediate representations
        p_emb = model.pitch_emb(ctx_p)
        s_emb = model.string_emb(ctx_s)
        f_emb = model.fret_emb(ctx_f)
        d_emb = model.duration_emb(ctx_d)
        i_emb = model.interval_emb(ctx_i)
        
        x = torch.cat([p_emb, s_emb, f_emb, d_emb, i_emb], dim=-1)
        x = model.input_proj(x)
        x = model.pos_encoding(x)
        
        # Extract attention from each layer
        for layer in model.transformer.layers:
            # Self-attention with weights
            attn_out, attn_w = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=True
            )
            # attn_w: (batch, seq, seq)
            # We care about what the last position attends to
            last_pos_attn = attn_w[:, -1, :].cpu().numpy()  # (batch, 16)
            position_importance += last_pos_attn.sum(axis=0)
            
            # Continue through the layer
            x = layer(x)
        
        n_batches += len(batch)

position_importance /= (n_batches * args["num_layers"])

print("\n  Attention weight by context position (position 16 = most recent):")
for pos in range(16):
    bar = "#" * int(position_importance[pos] * 100)
    print(f"    pos {pos+1:2d}: {position_importance[pos]:.4f} {bar}")

# Recency ratio
recent_4 = position_importance[-4:].sum()
early_12 = position_importance[:-4].sum()
print(f"\n  Recency bias: last 4 = {recent_4:.4f}, first 12 = {early_12:.4f}")
print(f"  Ratio (last4/first12): {recent_4/early_12:.2f}x")

# ========================================
# Part 2: Feature Ablation
# ========================================
print(f"\n{'='*60}")
print("Part 2: Feature Ablation (what matters for string choice?)")
print("=" * 60)

def eval_accuracy(samples, modify_fn=None):
    correct = 0
    total = 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start+batch_size]
        
        cp = torch.tensor([s["context_pitches"] for s in batch]).to(device)
        cs = torch.tensor([s["context_strings"] for s in batch]).to(device)
        cf = torch.tensor([[min(f,24) for f in s["context_frets"]] for s in batch]).to(device)
        cd = torch.tensor([s["context_durations"] for s in batch]).to(device)
        ci = torch.tensor([s["context_intervals"] for s in batch]).to(device)
        tp = torch.tensor([s["target_pitch"] for s in batch]).to(device)
        td = torch.tensor([s["target_duration"] for s in batch]).to(device)
        ti = torch.tensor([s["target_interval"] for s in batch]).to(device)
        pc = torch.tensor([s["position_context"] for s in batch]).to(device)
        gt = torch.tensor([s["target_string"] - 1 for s in batch]).to(device)
        
        if modify_fn:
            cp, cs, cf, cd, ci, tp, td, ti, pc = modify_fn(cp, cs, cf, cd, ci, tp, td, ti, pc)
        
        with torch.no_grad():
            logits = model(cp, cs, cf, cd, ci, tp, td, ti, pc)
            preds = logits.argmax(dim=-1)
            correct += (preds == gt).sum().item()
            total += len(batch)
    
    return correct / total

baseline = eval_accuracy(test_samples)
print(f"  Baseline (all features): {baseline:.4f}")

# Ablate each feature
ablations = {
    "zero context_strings": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, torch.zeros_like(cs), cf, cd, ci, tp, td, ti, pc),
    "zero context_frets": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, cs, torch.zeros_like(cf), cd, ci, tp, td, ti, pc),
    "zero context_durations": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, cs, cf, torch.zeros_like(cd), ci, tp, td, ti, pc),
    "zero context_intervals": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, cs, cf, cd, torch.zeros_like(ci), tp, td, ti, pc),
    "zero position_context": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, cs, cf, cd, ci, tp, td, ti, torch.zeros_like(pc)),
    "zero target_pitch": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, cs, cf, cd, ci, torch.zeros_like(tp), td, ti, pc),
    "shuffle context_strings": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (cp, cs[:, torch.randperm(cs.size(1))], cf, cd, ci, tp, td, ti, pc),
    "only last 4 context": lambda cp,cs,cf,cd,ci,tp,td,ti,pc: (
        torch.cat([torch.zeros_like(cp[:,:12]), cp[:,12:]], dim=1),
        torch.cat([torch.zeros_like(cs[:,:12]), cs[:,12:]], dim=1),
        torch.cat([torch.zeros_like(cf[:,:12]), cf[:,12:]], dim=1),
        cd, ci, tp, td, ti, pc
    ),
}

print(f"\n  {'Ablation':<30} {'Acc':>7} {'Drop':>7}")
print(f"  {'-'*45}")
for name, fn in ablations.items():
    acc = eval_accuracy(test_samples, fn)
    drop = baseline - acc
    print(f"  {name:<30} {acc:.4f} {drop:+.4f}")

# ========================================
# Part 3: Statistical Rule Extraction
# ========================================
print(f"\n{'='*60}")
print("Part 3: Statistical Rules from Training Data")
print("=" * 60)

# Load a subset of training data for analysis
rules_data = []
with open(DATA_DIR / "fingering_train.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 500000:
            break
        rules_data.append(json.loads(line))

print(f"  Analyzing {len(rules_data):,} samples...")

# Rule 1: String change vs pitch interval
interval_string_change = defaultdict(lambda: {"same": 0, "adj": 0, "far": 0})
for s in rules_data:
    prev_string = s["context_strings"][-1]
    cur_string = s["target_string"]
    interval = s["target_interval"] - 24  # back to -24..+24
    
    string_diff = abs(cur_string - prev_string)
    bucket = abs(interval)
    if bucket > 12:
        bucket = 13  # group >12
    
    key = bucket
    if string_diff == 0:
        interval_string_change[key]["same"] += 1
    elif string_diff == 1:
        interval_string_change[key]["adj"] += 1
    else:
        interval_string_change[key]["far"] += 1

print("\n  Rule 1: Pitch interval vs string change probability")
print(f"  {'Interval':>8} | {'Same':>6} {'Adj':>6} {'Far':>6} | {'Same%':>6}")
print(f"  {'-'*50}")
for iv in sorted(interval_string_change.keys()):
    d = interval_string_change[iv]
    total = d["same"] + d["adj"] + d["far"]
    pct_same = d["same"] / total * 100
    print(f"  {iv:>8} | {d['same']:>6} {d['adj']:>6} {d['far']:>6} | {pct_same:>5.1f}%")

# Rule 2: Position stickiness
pos_change = {"stay": 0, "move_small": 0, "move_big": 0}
for s in rules_data:
    prev_fret = s["context_frets"][-1]
    # Calculate current fret
    cur_string = s["target_string"]
    cur_fret = s["target_pitch"] - OPEN_PITCH.get(cur_string, 40)
    if 0 <= cur_fret <= 24:
        diff = abs(cur_fret - prev_fret)
        if diff <= 2:
            pos_change["stay"] += 1
        elif diff <= 5:
            pos_change["move_small"] += 1
        else:
            pos_change["move_big"] += 1

total_pos = sum(pos_change.values())
print(f"\n  Rule 2: Position stickiness")
for k, v in pos_change.items():
    print(f"    {k}: {v:,} ({v/total_pos*100:.1f}%)")

# Rule 3: String preference by pitch range
pitch_string_pref = defaultdict(Counter)
for s in rules_data:
    pitch = s["target_pitch"]
    string = s["target_string"]
    # Group pitch into ranges
    if pitch < 45:
        pr = "low (<E2)"
    elif pitch < 52:
        pr = "mid-low (E2-E3)"
    elif pitch < 60:
        pr = "middle (E3-B3)"
    elif pitch < 67:
        pr = "mid-high (C4-F#4)"
    else:
        pr = "high (>=G4)"
    pitch_string_pref[pr][string] += 1

print(f"\n  Rule 3: Preferred string by pitch range")
for pr in ["low (<E2)", "mid-low (E2-E3)", "middle (E3-B3)", "mid-high (C4-F#4)", "high (>=G4)"]:
    if pr in pitch_string_pref:
        c = pitch_string_pref[pr]
        total = sum(c.values())
        top = c.most_common(3)
        top_str = ", ".join([f"str{s}={n/total*100:.0f}%" for s, n in top])
        print(f"    {pr:<20}: {top_str}")

# Rule 4: Consecutive same-string runs
run_lengths = []
cur_run = 1
for s in rules_data:
    prev_s = s["context_strings"][-1]
    cur_s = s["target_string"]
    if cur_s == prev_s:
        cur_run += 1
    else:
        run_lengths.append(cur_run)
        cur_run = 1
run_lengths.append(cur_run)

run_counter = Counter()
for r in run_lengths:
    if r <= 10:
        run_counter[r] += 1
    else:
        run_counter[">10"] += 1

print(f"\n  Rule 4: Consecutive same-string run lengths")
total_runs = sum(run_counter.values())
for k in sorted([k for k in run_counter if isinstance(k, int)]):
    pct = run_counter[k] / total_runs * 100
    print(f"    {k} notes: {run_counter[k]:,} ({pct:.1f}%)")
if ">10" in run_counter:
    print(f"    >10 notes: {run_counter['>10']:,} ({run_counter['>10']/total_runs*100:.1f}%)")

print(f"\n  Mean run length: {np.mean(run_lengths):.2f}")
print(f"  Median run length: {np.median(run_lengths):.1f}")

print("\n=== Analysis Complete ===")
