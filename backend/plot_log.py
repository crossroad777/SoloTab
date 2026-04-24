import os
import sys
import re
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs = []
    train_loss = []
    val_loss = []
    mpe_f1 = []
    tdr_f1 = []
    onset_f1 = []

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex patterns
    # --- Epoka 1/300 ---
    #   LR: 1.00e-05 | Train Loss: 2.3773 | Val Loss: 2.1605
    #   MPE F1 (ramki): 0.7146
    #   TDR F1 (nuty, thr=0.5): 0.4002 (P: 0.4218, R: 0.3998)
    #   Onset F1 (event, thr=0.5): 0.6077

    epoch_blocks = content.split('--- Epoka ')
    for block in epoch_blocks[1:]:
        try:
            ep_match = re.search(r'^(\d+)/\d+ ---', block)
            if not ep_match: continue
            ep = int(ep_match.group(1))
            
            # Loss
            loss_m = re.search(r'Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', block)
            if not loss_m: continue
            tl, vl = float(loss_m.group(1)), float(loss_m.group(2))
            
            # MPE
            mpe_m = re.search(r'MPE F1.*?([\d.]+)', block)
            mf1 = float(mpe_m.group(1)) if mpe_m else 0.0
            
            # TDR F1
            tdr_m = re.search(r'TDR F1.*?([\d.]+)', block)
            tf1 = float(tdr_m.group(1)) if tdr_m else 0.0
            
            # Onset F1
            onset_m = re.search(r'Onset F1.*?([\d.]+)', block)
            of1 = float(onset_m.group(1)) if onset_m else 0.0
            
            epochs.append(ep)
            train_loss.append(tl)
            val_loss.append(vl)
            mpe_f1.append(mf1)
            tdr_f1.append(tf1)
            onset_f1.append(of1)
        except Exception as e:
            pass

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'mpe_f1': mpe_f1,
        'tdr_f1': tdr_f1,
        'onset_f1': onset_f1
    }

def plot_history(data, save_path):
    if not data['epochs']:
        print("No valid epoch data found.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axs[0].plot(data['epochs'], data['train_loss'], 'o-', label='Train Loss')
    axs[0].plot(data['epochs'], data['val_loss'], 'o-', label='Validation Loss')
    axs[0].set_title('Training & Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()
    
    # F1 plot
    axs[1].plot(data['epochs'], data['tdr_f1'], 'o-', label='Note F1 (TDR)')
    axs[1].plot(data['epochs'], data['onset_f1'], 's-', label='Onset F1')
    axs[1].plot(data['epochs'], data['mpe_f1'], '^-', label='Frame F1 (MPE)')
    axs[1].set_title('F1 Scores')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('F1 Score')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_log.py path/to/training_log.txt")
        sys.exit(1)
        
    log_file = sys.argv[1]
    save_file = log_file.replace('.txt', '_plot.png')
    
    data = parse_log(log_file)
    plot_history(data, save_file)
