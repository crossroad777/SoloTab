import os
import sys
import re
import matplotlib.pyplot as plt

project_root = r"D:\Music\nextchord-solotab"
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
sys.path.insert(0, mt_python_dir)

# The log file is inside finetuned_martin_model
log_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "finetuned_martin_model", "training_log.txt")

history = {
    "epochs": [],
    "train_total_loss": [],
    "val_total_loss": [],
    "lr": [],
    "val_tdr_f1_at_0.5": [],
    "val_mpe_f1": [],
    "val_onset_f1_event_at_0.5": []
}

if not os.path.exists(log_path):
    print(f"Błąd: Nie znaleziono pliku {log_path} - być może epoka 1 jeszcze się nie zakończyła.")
    sys.exit(1)

with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

# Parse the text blocks
# --- Epoka 1/300 ---
#   LR: 1.00e-04 | Train Loss: 0.1234 | Val Loss: 0.1200
#   MPE F1 (ramki): 0.8210
#   TDR F1 (nuty, thr=0.5): 0.8123 (P: 0.8000, R: 0.8200)
#   Onset F1 (event, thr=0.5): 0.8300

epoch_blocks = text.split("--- Epoka ")
for block in epoch_blocks[1:]:
    try:
        epoch_str = block.split("/")[0]
        epoch = int(epoch_str)
        
        lr_match = re.search(r"LR:\s*([0-9\.e\-]+)", block)
        tl_match = re.search(r"Train Loss:\s*([0-9\.]+)", block)
        vl_match = re.search(r"Val Loss:\s*([0-9\.]+)", block)
        mpe_match = re.search(r"MPE F1 \(ramki\):\s*([0-9\.]+)", block)
        tdr_match = re.search(r"TDR F1 \(nuty, thr=0.5\):\s*([0-9\.]+)\s+\(P:\s*([0-9\.]+),\s*R:\s*([0-9\.]+)\)", block)
        onset_match = re.search(r"Onset F1 \(event, thr=0.5\):\s*([0-9\.]+)", block)
        
        if tl_match and tdr_match:
            history["epochs"].append(epoch)
            history["lr"].append(float(lr_match.group(1)) if lr_match else 0.0001)
            history["train_total_loss"].append(float(tl_match.group(1)))
            history["val_total_loss"].append(float(vl_match.group(1)) if vl_match else 0.0)
            history["val_mpe_f1"].append(float(mpe_match.group(1)) if mpe_match else 0.0)
            history["val_tdr_f1_at_0.5"].append(float(tdr_match.group(1)))
            history.setdefault("val_tdr_precision_at_0.5", []).append(float(tdr_match.group(2)))
            history.setdefault("val_tdr_recall_at_0.5", []).append(float(tdr_match.group(3)))
            history["val_onset_f1_event_at_0.5"].append(float(onset_match.group(1)) if onset_match else 0.0)
    except Exception as e:
        continue

if not history["epochs"]:
    print("Nie znaleziono jeszcze pełnych wyników w logach. Epoka 1 musi się zakończyć.")
    sys.exit(0)

# Import internal plotting
from vizualization import plotting
save_img_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", "finetuned_martin_model", "training_progress.png")

plotting.plot_training_history(history, output_save_path=save_img_path)
print(f"\nWykres wygenerowany pomyślnie!")
print(f"Sprawdź: {save_img_path}")
