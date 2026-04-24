import os
import sys
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

SPACE_ID = "Crossroad777/solotab"
BASE_DIR = Path(r"D:\Music\nextchord-solotab")
STAGE_DIR = Path(r"D:\Music\nextchord-solotab\_hf_stage")

# Clean and create staging dir
if STAGE_DIR.exists():
    shutil.rmtree(STAGE_DIR)
STAGE_DIR.mkdir(parents=True)

print("=== Step 1: Staging backend files ===")
backend_dir = STAGE_DIR / "backend"
backend_dir.mkdir()

# Copy backend Python files (no __pycache__, no training scripts)
backend_src = BASE_DIR / "backend"
skip_patterns = [
    "train_", "finetune_", "prepare_", "precache_", "debug_",
    "augment_", "convert_", "download_", "test_",
    "__pycache__", ".bat", "analyze_", "gp5_"
]
for f in backend_src.iterdir():
    if f.is_file() and f.suffix == ".py":
        skip = any(f.name.startswith(p) or f.name.endswith(p) for p in skip_patterns)
        if not skip:
            shutil.copy2(f, backend_dir / f.name)
            print(f"  Copied: {f.name}")

# Copy requirements
if (backend_src / "requirements.txt").exists():
    shutil.copy2(backend_src / "requirements.txt", backend_dir / "requirements.txt")

print("\n=== Step 2: Staging frontend files ===")
frontend_src = BASE_DIR / "frontend"
frontend_dst = STAGE_DIR / "frontend"

# Copy frontend source (exclude node_modules)
def ignore_patterns(d, files):
    return [f for f in files if f in ("node_modules", ".git", "dist")]

shutil.copytree(frontend_src, frontend_dst, ignore=ignore_patterns)
print("  Frontend copied (excluding node_modules)")

print("\n=== Step 3: Staging model files ===")
models_dir = STAGE_DIR / "models"
models_dir.mkdir()

# FretNet fold models
fretnet_src = BASE_DIR / "generated" / "fretnet_models" / "models"
if fretnet_src.exists():
    for fold_dir in sorted(fretnet_src.iterdir()):
        if fold_dir.is_dir() and fold_dir.name.startswith("fold-"):
            dst_fold = models_dir / "fretnet" / fold_dir.name
            dst_fold.mkdir(parents=True, exist_ok=True)
            # Find best model
            pt_files = list(fold_dir.glob("*.pt"))
            if pt_files:
                best = sorted(pt_files, key=lambda x: x.stat().st_mtime)[-1]
                shutil.copy2(best, dst_fold / "model.pt")
                print(f"  FretNet {fold_dir.name}: {best.name} ({best.stat().st_size/1024/1024:.1f} MB)")

# FretNet scoreset finetune
scoreset_dir = BASE_DIR / "generated" / "fretnet_scoreset_finetune" / "models" / "scoreset-finetune"
if scoreset_dir.exists():
    dst = models_dir / "fretnet" / "scoreset-finetune"
    dst.mkdir(parents=True, exist_ok=True)
    pt_files = list(scoreset_dir.glob("*.pt"))
    if pt_files:
        best = sorted(pt_files, key=lambda x: x.stat().st_mtime)[-1]
        shutil.copy2(best, dst / "model.pt")
        print(f"  FretNet scoreset: {best.name} ({best.stat().st_size/1024/1024:.1f} MB)")

# SynthTab
synthtab_dirs = list((BASE_DIR / "generated" / "synthtab_finetune").glob("models_*"))
if synthtab_dirs:
    latest_dir = sorted(synthtab_dirs)[-1]
    pt_files = list(latest_dir.glob("*.pt"))
    opt_files = [f for f in pt_files if "opt-state" in f.name]
    model_files = [f for f in pt_files if "opt-state" not in f.name]
    if model_files:
        best = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
        dst = models_dir / "synthtab"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst / "model.pt")
        print(f"  SynthTab: {best.name} ({best.stat().st_size/1024/1024:.1f} MB)")

# SynthTab pretrained
pretrained = BASE_DIR / "music-transcription" / "SynthTab" / "demo_embedding" / "pretrained_models" / "SynthTab-Pretrained.pt"
if pretrained.exists():
    dst = models_dir / "synthtab"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pretrained, dst / "SynthTab-Pretrained.pt")
    print(f"  SynthTab pretrained: ({pretrained.stat().st_size/1024/1024:.1f} MB)")

# CRNN
crnn_dirs = list((BASE_DIR / "music-transcription" / "python").glob("outputs/*/best_model.pth"))
if crnn_dirs:
    best = sorted(crnn_dirs, key=lambda x: x.stat().st_mtime)[-1]
    dst = models_dir / "crnn"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst / "best_model.pth")
    print(f"  CRNN: ({best.stat().st_size/1024/1024:.1f} MB)")

# Technique classifier
tech_model = BASE_DIR / "generated" / "technique_classifier" / "best_model.pt"
if tech_model.exists():
    dst = models_dir / "technique_classifier"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tech_model, dst / "best_model.pt")
    print(f"  Technique classifier: ({tech_model.stat().st_size/1024/1024:.1f} MB)")

# String classifier
str_model = BASE_DIR / "generated" / "string_classifier" / "best_model.pt"
if str_model.exists():
    dst = models_dir / "string_classifier"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str_model, dst / "best_model.pt")
    print(f"  String classifier: ({str_model.stat().st_size/1024/1024:.1f} MB)")

# SynthTab code (needed for model definitions)
synthtab_code = BASE_DIR / "music-transcription" / "SynthTab"
if synthtab_code.exists():
    # Copy only the Python source files needed
    dst = STAGE_DIR / "music-transcription" / "SynthTab"
    for subdir in ["amt", "demo_embedding"]:
        src_sub = synthtab_code / subdir
        if src_sub.exists():
            shutil.copytree(src_sub, dst / subdir, ignore=lambda d, f: [x for x in f if x in ("__pycache__", "pretrained_models")])

# FretNet code
fretnet_code = BASE_DIR / "music-transcription" / "FretNet"
if fretnet_code.exists():
    dst = STAGE_DIR / "music-transcription" / "FretNet"
    shutil.copytree(fretnet_code, dst, ignore=lambda d, f: [x for x in f if x in ("__pycache__", "exp", "data")])

# CRNN code
crnn_code = BASE_DIR / "music-transcription" / "python"
if crnn_code.exists():
    dst = STAGE_DIR / "music-transcription" / "python"
    dst.mkdir(parents=True, exist_ok=True)
    for f in crnn_code.iterdir():
        if f.is_file() and f.suffix == ".py":
            shutil.copy2(f, dst / f.name)

print("\n=== Step 4: Calculate staging size ===")
total = 0
for f in STAGE_DIR.rglob("*"):
    if f.is_file():
        total += f.stat().st_size
print(f"Total staging size: {total/1024/1024:.1f} MB")

print("\n=== Staging complete! ===")
