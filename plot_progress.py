import re
import matplotlib.pyplot as plt
import os

log_file = r"D:\Music\nextchord-solotab\verify_train21.log"
output_image = r"C:\Users\kotan\.gemini\antigravity\brain\f5c4e0c4-1078-42b5-8421-907a1600d6e9\learning_curve.png"

epochs = []
train_losses = []
val_losses = []

if not os.path.exists(log_file):
    print("Log file not found.")
    exit(1)

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Extract epochs: --- Epoka X/300 ---
# Followed by:   LR: 1.00e-03 | Train Loss: 0.1234 | Val Loss: 0.1234
epoch_matches = re.finditer(r"--- Epoka (\d+)/300 ---.*?Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+)", text, re.DOTALL)

for m in epoch_matches:
    ep = int(m.group(1))
    tl = float(m.group(2))
    vl = float(m.group(3))
    epochs.append(ep)
    train_losses.append(tl)
    val_losses.append(vl)

if not epochs:
    print("No complete epochs found in log yet.")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linewidth=2, marker='s', markersize=4)
plt.title(f"Acoustic Mixed Model Fine-Tuning Progress (Up to Epoch {epochs[-1]})", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (CrossEntropy, BCE, Fret)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_image, dpi=150)
print(f"Successfully generated plot at {output_image}")
