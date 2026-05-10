"""
train_fingering_v3.py — V3: Transformer + 拡張特徴量での学習
============================================================
LSTMv3 と Transformer の両方を比較実験可能。
"""
import sys, json, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from fingering_model_v3 import FingeringTransformer, FingeringLSTMv3

DATA_DIR = Path(__file__).parent.parent.parent / "gp_training_data" / "v3"
MODEL_DIR = DATA_DIR / "models"


class FingeringDatasetV3(Dataset):
    def __init__(self, filepath, max_samples=None):
        self.samples = []
        print(f"  Loading {filepath.name}...", end=" ", flush=True)
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        print(f"{len(self.samples):,} loaded")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "context_pitches": torch.tensor(s["context_pitches"], dtype=torch.long),
            "context_strings": torch.tensor(s["context_strings"], dtype=torch.long),
            "context_frets": torch.tensor(
                [min(f, 24) for f in s["context_frets"]], dtype=torch.long
            ),
            "context_durations": torch.tensor(s["context_durations"], dtype=torch.long),
            "context_intervals": torch.tensor(s["context_intervals"], dtype=torch.long),
            "target_pitch": torch.tensor(s["target_pitch"], dtype=torch.long),
            "target_string": torch.tensor(s["target_string"] - 1, dtype=torch.long),
            "target_duration": torch.tensor(s["target_duration"], dtype=torch.long),
            "target_interval": torch.tensor(s["target_interval"], dtype=torch.long),
            "position_context": torch.tensor(s["position_context"], dtype=torch.long),
            "num_candidates": torch.tensor(s["num_candidates"], dtype=torch.long),
            "is_ambiguous": torch.tensor(s["is_ambiguous"], dtype=torch.bool),
        }


def forward_batch(model, batch, device):
    """モデルの forward を呼ぶ共通関数"""
    return model(
        batch["context_pitches"].to(device),
        batch["context_strings"].to(device),
        batch["context_frets"].to(device),
        batch["context_durations"].to(device),
        batch["context_intervals"].to(device),
        batch["target_pitch"].to(device),
        batch["target_duration"].to(device),
        batch["target_interval"].to(device),
        batch["position_context"].to(device),
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        tgt_s = batch["target_string"].to(device)
        optimizer.zero_grad()
        logits = forward_batch(model, batch, device)
        loss = criterion(logits, tgt_s)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * tgt_s.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == tgt_s).sum().item()
        total += tgt_s.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_detailed(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_all = 0; total_all = 0
    correct_amb = 0; total_amb = 0
    string_correct = [0]*6; string_total = [0]*6
    cand_correct = {}; cand_total = {}
    
    for batch in loader:
        tgt_s = batch["target_string"].to(device)
        is_amb = batch["is_ambiguous"].to(device)
        num_cand = batch["num_candidates"]
        
        logits = forward_batch(model, batch, device)
        loss = criterion(logits, tgt_s)
        total_loss += loss.item() * tgt_s.size(0)
        
        pred = logits.argmax(dim=-1)
        hit = (pred == tgt_s)
        
        correct_all += hit.sum().item()
        total_all += tgt_s.size(0)
        correct_amb += (hit & is_amb).sum().item()
        total_amb += is_amb.sum().item()
        
        for s in range(6):
            mask = (tgt_s == s)
            string_total[s] += mask.sum().item()
            string_correct[s] += (hit & mask).sum().item()
        
        for nc in num_cand.unique().tolist():
            mask = (num_cand == nc)
            key = int(nc)
            cand_total[key] = cand_total.get(key, 0) + mask.sum().item()
            cand_correct[key] = cand_correct.get(key, 0) + (hit.cpu() & mask).sum().item()
    
    result = {
        "loss": total_loss / total_all,
        "acc_all": correct_all / total_all,
        "acc_ambiguous": correct_amb / max(1, total_amb),
        "string_acc": {f"str{s+1}": string_correct[s]/max(1,string_total[s]) for s in range(6)},
        "candidate_acc": {f"{nc}_cand": cand_correct[nc]/max(1,cand_total[nc]) for nc in sorted(cand_total)},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["transformer", "lstm"], default="transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Arch: {args.arch}")
    
    print(f"\n=== Loading V3 Data ===")
    train_ds = FingeringDatasetV3(DATA_DIR / "fingering_train.jsonl", args.max_train)
    val_ds = FingeringDatasetV3(DATA_DIR / "fingering_val.jsonl", args.max_val)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            pin_memory=True, num_workers=0)
    
    if args.arch == "transformer":
        model = FingeringTransformer(
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, embed_dim=args.embed_dim,
            dropout=0.1,
        ).to(device)
    else:
        model = FingeringLSTMv3(
            embed_dim=args.embed_dim * 2, hidden_dim=args.d_model * 2,
            num_layers=args.num_layers, dropout=0.2,
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch}, {total_params:,} params")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc_amb = 0
    model_name = f"fingering_{args.arch}_v3_best.pt"
    
    print(f"\n=== Training V3 ({args.arch}) ===")
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    print(f"-" * 85)
    print(f"{'Ep':>3} | {'TrLoss':>7} {'TrAcc':>7} | "
          f"{'VaLoss':>7} {'VaAll':>7} {'VaAmb':>7} | {'Time':>5} {'Note':>6}")
    print(f"-" * 85)
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate_detailed(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        is_best = val_result["acc_ambiguous"] > best_val_acc_amb
        if is_best:
            best_val_acc_amb = val_result["acc_ambiguous"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_result": val_result,
                "args": vars(args),
                "arch": args.arch,
            }, MODEL_DIR / model_name)
        
        mark = "*BEST*" if is_best else ""
        print(f"{epoch:3d} | {train_loss:7.4f} {train_acc:7.4f} | "
              f"{val_result['loss']:7.4f} {val_result['acc_all']:7.4f} "
              f"{val_result['acc_ambiguous']:7.4f} | {elapsed:5.0f}s {mark}")
        
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"    String: ", end="")
            for s, a in val_result["string_acc"].items():
                print(f"{s}={a:.3f} ", end="")
            print()
            print(f"    Cand:   ", end="")
            for c, a in val_result["candidate_acc"].items():
                print(f"{c}={a:.3f} ", end="")
            print()
    
    # Test
    print(f"\n{'='*85}")
    print(f"Best ambiguous acc: {best_val_acc_amb:.4f}")
    print(f"\n=== Test ===")
    test_ds = FingeringDatasetV3(DATA_DIR / "fingering_test.jsonl", args.max_val)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, pin_memory=True)
    
    ckpt = torch.load(MODEL_DIR / model_name, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_r = evaluate_detailed(model, test_loader, criterion, device)
    print(f"Test ALL:       {test_r['acc_all']:.4f}")
    print(f"Test AMBIGUOUS: {test_r['acc_ambiguous']:.4f}")
    print(f"String: {test_r['string_acc']}")
    print(f"Cand:   {test_r['candidate_acc']}")


if __name__ == "__main__":
    main()
