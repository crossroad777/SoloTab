import os
import shutil
import glob

# 必須ファイル群（絶対に移動しない）
KEEP_FILES = {
    "main.py",
    "pipeline.py",
    "pure_moe_transcriber.py",
    "guitar_transcriber.py",
    "tab_renderer.py",
    "chord_detector.py",
    "technique_detector.py",
    "tuning_detector.py",
    "beat_detector.py",
    "capo_detector.py",
    "key_analyzer.py",
    "solotab_utils.py",
    "config.py"
}

# カテゴリ別の退避フォルダ
DIRS = {
    "benchmark": "benchmark",
    "train": "train",
    "tools": "tools",
    "synth": "synth",
    "legacy": "legacy",
    "logs": "logs"
}

# フォルダの作成
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# 拡張子や接頭辞ベースで分類
for item in os.listdir("."):
    if os.path.isdir(item) or item == os.path.basename(__file__):
        continue
    
    if item in KEEP_FILES:
        continue
        
    # 分類ロジック
    dest_dir = None
    if item.endswith(".log") or item.endswith(".txt") or item.endswith(".json"):
        if item != "requirements.txt": # requirements.txt はキープ
            dest_dir = "logs"
    elif item.startswith("benchmark_"):
        dest_dir = "benchmark"
    elif item.startswith("train_") or item.startswith("resume_") or item.startswith("simulate_") or item.startswith("tune_"):
        dest_dir = "train"
    elif item.startswith("analyze_") or item.startswith("convert_") or item.startswith("debug_") or item.startswith("test_") or item.startswith("profile_") or item.startswith("tmp_") or item.startswith("check_") or item.startswith("download_") or item.startswith("plot_") or item.startswith("read_"):
        dest_dir = "tools"
    elif item.startswith("synth_") or item.startswith("generate_") or item == "guitar_synth.py" or item == "karplus_strong.py" or item == "synthtab_midi_synthesizer.py" or item == "dadagp_tokenizer.py":
        dest_dir = "synth"
    elif item.endswith(".py") or item.endswith(".ps1"):
        # 上記に当てはまらないものは legacy（旧処理モジュールやテスト用の残骸など）へ
        dest_dir = "legacy"
        
    if dest_dir:
        shutil.move(item, os.path.join(dest_dir, item))

print("ファイルの整理が完了しました。")
