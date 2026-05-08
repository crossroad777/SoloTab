"""
run_synth_experiment.py — 合成データ効果検証実験
================================================
martin_finger の GA Best (F1=0.7927) をソースモデルとして、
GuitarSet + GAPS + Synth V2 (500件) で10エポック学習し、
合成データの有無でF1に差が出るか検証する。
"""
import os, sys

os.chdir(r"d:\Music\nextchord-solotab\music-transcription\python")
sys.path.insert(0, r"d:\Music\nextchord-solotab")
sys.path.insert(0, r"d:\Music\nextchord-solotab\music-transcription\python")

from backend.train.train_gaps_multitask import finetune_multitask, TRAINING_OUTPUT_DIR
import torch, shutil

# Experiment setup
DOMAIN = "martin_finger"
EPOCHS = 10
PATIENCE = 10

# Reset the GA log so it doesn't skip
ga_dir = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{DOMAIN}_multitask_3ds_ga")
log_path = os.path.join(ga_dir, "training_log.txt")
best_path = os.path.join(ga_dir, "best_model.pth")

# Backup current best model and log
backup_dir = os.path.join(ga_dir, "_backup_pre_synth_experiment")
os.makedirs(backup_dir, exist_ok=True)
if os.path.exists(best_path) and not os.path.exists(os.path.join(backup_dir, "best_model.pth")):
    shutil.copy2(best_path, os.path.join(backup_dir, "best_model.pth"))
if os.path.exists(log_path):
    shutil.copy2(log_path, os.path.join(backup_dir, "training_log.txt"))

# Clear log to prevent "already early-stopped" skip
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"--- Synth V2 Experiment | martin_finger ---\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiment: {DOMAIN} + Synth V2 (500), {EPOCHS} epochs")
print(f"Source model: {best_path}")
print(f"Baseline F1: 0.7927 (GA Best without synth)")

finetune_multitask(DOMAIN, device, ft_epochs=EPOCHS, ft_patience=PATIENCE,
                   include_agpt=True, ga_retrain=True)
