"""
train_string_predictor.py — MLM弦予測Transformerの学習
=====================================================
DadaGPデータセットでTransformer MLMを事前学習し、
ギターノートの弦割り当てを学習する。

Usage:
    python train_string_predictor.py --token-dir ../datasets/dadaGP/examples
    python train_string_predictor.py --token-dir D:/datasets/dadaGP_tokens --epochs 50
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from string_predictor_model import StringPredictor, create_mlm_mask
from dadagp_tokenizer import DadaGPDataset, collate_fn


def train_epoch(model, loader, optimizer, scheduler, device, 
                mask_ratio=0.15, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in loader:
        pitches = batch['pitches'].to(device)
        time_shifts = batch['time_shifts'].to(device)
        durations = batch['durations'].to(device)
        strings = batch['strings'].to(device)
        lengths = batch['lengths'].to(device)
        
        B, L = pitches.shape
        
        # Create padding mask
        padding_mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Create MLM mask
        mlm_mask = create_mlm_mask(strings, mask_ratio=mask_ratio)
        
        # Forward pass
        logits = model(pitches, time_shifts, durations, padding_mask)
        
        # Compute loss only on masked positions
        masked_logits = logits[mlm_mask]  # (N_masked, 6)
        masked_targets = strings[mlm_mask]  # (N_masked,)
        
        if masked_logits.shape[0] == 0:
            continue
        
        loss = F.cross_entropy(masked_logits, masked_targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item() * masked_logits.shape[0]
        predictions = masked_logits.argmax(dim=-1)
        total_correct += (predictions == masked_targets).sum().item()
        total_masked += masked_logits.shape[0]
    
    avg_loss = total_loss / max(total_masked, 1)
    accuracy = total_correct / max(total_masked, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device, mask_ratio=1.0):
    """Evaluate model on validation set.
    
    For evaluation, mask ALL strings (mask_ratio=1.0) to simulate inference.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_valid = 0
    
    for batch in loader:
        pitches = batch['pitches'].to(device)
        time_shifts = batch['time_shifts'].to(device)
        durations = batch['durations'].to(device)
        strings = batch['strings'].to(device)
        lengths = batch['lengths'].to(device)
        
        B, L = pitches.shape
        padding_mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        logits = model(pitches, time_shifts, durations, padding_mask)
        
        # Compute on all valid (non-padded) positions
        valid_mask = strings >= 0
        valid_logits = logits[valid_mask]
        valid_targets = strings[valid_mask]
        
        if valid_logits.shape[0] == 0:
            continue
        
        loss = F.cross_entropy(valid_logits, valid_targets)
        predictions = valid_logits.argmax(dim=-1)
        
        total_loss += loss.item() * valid_logits.shape[0]
        total_correct += (predictions == valid_targets).sum().item()
        total_valid += valid_logits.shape[0]
    
    avg_loss = total_loss / max(total_valid, 1)
    accuracy = total_correct / max(total_valid, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train MLM String Predictor on DadaGP or Synthetic Data'
    )
    parser.add_argument('--token-dir', type=str, default=None,
                        help='Directory with .tokens.txt files (DadaGP mode)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic training data instead of DadaGP')
    parser.add_argument('--num-samples', type=int, default=50000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join('..', 'generated', 'string_predictor'),
                        help='Output directory for model checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--max-files', type=int, default=None,
                        help='Limit number of token files (for debugging)')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--mask-ratio', type=float, default=0.15)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=5)
    args = parser.parse_args()
    
    if not args.synthetic and not args.token_dir:
        print("ERROR: Either --token-dir or --synthetic is required")
        sys.exit(1)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    if args.synthetic:
        from generate_synthetic_tab_data import SyntheticTabDataset
        dataset = SyntheticTabDataset(
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
        )
    else:
        dataset = DadaGPDataset(
            token_dir=args.token_dir,
            max_seq_len=args.max_seq_len,
            max_files=args.max_files,
        )
    
    if len(dataset) == 0:
        print("ERROR: No training data!")
        sys.exit(1)
    
    # Train/Val split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    
    print(f"Train: {n_train} chunks, Val: {n_val} chunks")
    
    # Model
    model = StringPredictor(
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    
    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f"\nTraining: {args.epochs} epochs, LR={args.lr}, BS={args.batch_size}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Output: {args.output_dir}\n")
    
    best_val_acc = 0.0
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            mask_ratio=args.mask_ratio,
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        elapsed = time.time() - t0
        eta = elapsed / epoch * (args.epochs - epoch)
        lr_now = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"lr={lr_now:.2e} elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args),
            }, best_path)
            print(f"    ★ Best model saved (val_acc={val_acc:.3f})")
        
        # Periodic save
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'model_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, ckpt_path)
    
    # Final save
    final_path = os.path.join(args.output_dir, 'model_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'args': vars(args),
    }, final_path)
    
    elapsed = time.time() - t0
    print(f"\nTraining complete! {elapsed/60:.1f}min")
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
