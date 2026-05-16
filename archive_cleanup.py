"""
archive_cleanup.py — v1.0基準点コードベース整理スクリプト
=====================================================
不要なファイルを legacy_archive/v1.0_cleanup/ に移動し、
プロダクションコードを明確にする。

実行前に必ず git commit すること。
"""
import os
import shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(ROOT, "legacy_archive", "v1.0_cleanup")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# === 移動対象リスト ===

# 1. backend/ 内のデバッグ・一回限りスクリプト
BACKEND_DEBUG = [
    "debug_bass_full.py", "debug_boundary.py", "debug_cut.py", "debug_cut2.py",
    "debug_gp5_rhythm.py", "debug_pipeline.py", "debug_pipeline2.py",
    "debug_pos.py", "debug_timing.py", "debug_voices.py",
    "_fix_session.py", "check_beats.py", "check_dupes.py",
    "compare_pitch.py", "compare_reference.py",
    "dump_gp5.py", "e2e_test.py", "full_verify.py",
    "quick_benchmark.py", "regen_gp5.py", "render_tab.py",
    "test_render.py", "test_versatility.py", "trace_duration.py",
    "validate_gp_downloads.py", "verify_all.py", "verify_fixed.py", "verify_gp5.py",
]

# 2. backend/ 内の旧Optuna最適化スクリプト
BACKEND_OPTUNA = [
    "run_optuna_pima.py", "run_optuna_v2.py",
    "optimize_string_assignment.py",
]

# 3. backend/ 内の分析・ユーティリティ（本番未使用）
BACKEND_ANALYSIS = [
    "analyze_gaps.py", "analyze_gp5.py", "analyze_user_gp5.py",
    "tab_data_api.py",  # API未使用
    "musescore_renderer.py",  # MuseScore連携は廃止済み
    "pure_moe_transcriber.py",  # pipeline.pyに統合済み
]

# 4. backend/ 内のバックアップ弦分類器モデル（string_classifier.pthのみ残す）
BACKEND_MODEL_BACKUPS = [
    "string_classifier_3ds.pth",
    "string_classifier_augmented.pth",
    "string_classifier_generalized.pth",
    "string_classifier_step2_backup.pth",
]

# 5. backend/ 旧最適化重み
BACKEND_OLD_WEIGHTS = [
    "optimized_weights_pima.json",
    "optimized_weights_v2.json",
]

# 6. ルートの旧実験スクリプト
ROOT_EXPERIMENT = [
    "analyze_jams.py", "compare_cqt.py", "preprocess_idmt.py",
    "run_2stage_experiment.py", "run_all_7domains.py",
    "run_ga_retrain.py", "run_idmt_7domains.py", "run_synth_experiment.py",
    "benchmark_35model_moe.py", "benchmark_42model_moe.py",
    "benchmark_param_comparison.py", "benchmark_sf_ablation.py",
]

# 7. ルートの旧ログ・デモファイル
ROOT_LOGS = [
    "synth_v2_7domain_run.log", "synth_v2_forced_moe.log",
    "demo_fluidr3_test.wav", "demo_fluidsynth_guitar.wav",
    "demo_karplus_strong.wav",
]

# 8. scratch/ 内の旧ファイル
SCRATCH_OLD = [
    "old_benchmark.py", "old_string_assigner.py",
    "test_alignment.html", "test_alignment2.html",
    "test_beat_diag.py", "test_data_aug.py",
]

def move_files(file_list, source_dir, category):
    """ファイルをアーカイブに移動"""
    cat_dir = os.path.join(ARCHIVE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    moved = 0
    for f in file_list:
        src = os.path.join(source_dir, f)
        dst = os.path.join(cat_dir, f)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  [MOVED] {os.path.relpath(src, ROOT)} -> legacy_archive/v1.0_cleanup/{category}/{f}")
            moved += 1
        else:
            print(f"  [SKIP]  {f} (not found)")
    return moved

if __name__ == "__main__":
    print(f"=== SoloTab v1.0 Baseline Cleanup ===")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Archive: {ARCHIVE_DIR}")
    print()

    total = 0

    print("--- 1. Backend debug scripts ---")
    total += move_files(BACKEND_DEBUG, os.path.join(ROOT, "backend"), "debug_scripts")

    print("\n--- 2. Backend Optuna scripts ---")
    total += move_files(BACKEND_OPTUNA, os.path.join(ROOT, "backend"), "optuna_scripts")

    print("\n--- 3. Backend analysis/utility ---")
    total += move_files(BACKEND_ANALYSIS, os.path.join(ROOT, "backend"), "analysis_utils")

    print("\n--- 4. Backend model backups ---")
    total += move_files(BACKEND_MODEL_BACKUPS, os.path.join(ROOT, "backend"), "model_backups")

    print("\n--- 5. Backend old weights ---")
    total += move_files(BACKEND_OLD_WEIGHTS, os.path.join(ROOT, "backend"), "old_weights")

    print("\n--- 6. Root experiment scripts ---")
    total += move_files(ROOT_EXPERIMENT, ROOT, "root_experiments")

    print("\n--- 7. Root logs/demos ---")
    total += move_files(ROOT_LOGS, ROOT, "logs_demos")

    print("\n--- 8. Scratch old files ---")
    total += move_files(SCRATCH_OLD, os.path.join(ROOT, "scratch"), "scratch_old")

    print(f"\n=== Complete: {total} files archived ===")
