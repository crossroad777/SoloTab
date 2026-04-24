import sys
import os
import torch
import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

project_root = r"D:\Music\nextchord-solotab"
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
sys.path.insert(0, mt_python_dir)

print("\n--- 10-TIME ABSOLUTE VERIFICATION LOOP ---")

for i in range(1, 11):
    print(f"\n[Validation Run #{i}/10]")
    all_passed = True
    
    # 1. Check inference (guitar_transcriber.py)
    try:
        transcriber = load_module_from_path("guitar_transcriber", os.path.join(project_root, "backend", "guitar_transcriber.py"))
        model_path, config_path = transcriber._get_model_paths()
        if "baseline_model" not in model_path:
            print("  [FAIL] guitar_transcriber.py path is NOT baseline_model")
            all_passed = False
        else:
            state_dict = torch.load(model_path, weights_only=True)
            print("  [OK] guitar_transcriber.py correctly points to baseline_model and weights are loadable.")
    except Exception as e:
        print(f"  [FAIL] guitar_transcriber.py crashed: {e}")
        all_passed = False

    # 2. Check resume_martin_training.py
    try:
        martin = load_module_from_path("resume_martin_training", os.path.join(project_root, "backend", "resume_martin_training.py"))
        if "baseline_model" not in martin.MODEL_LOAD_PATH:
            print("  [FAIL] resume_martin_training.py LOAD path is NOT baseline_model")
            all_passed = False
        elif "finetuned_martin_model" not in martin.MODEL_SAVE_PATH:
            print("  [FAIL] resume_martin_training.py SAVE path does not protect baseline!")
            all_passed = False
        else:
            print("  [OK] resume_martin_training.py correctly separates LOAD (baseline) and SAVE (finetuned_martin_model).")
    except Exception as e:
        print(f"  [FAIL] resume_martin_training.py crashed: {e}")
        all_passed = False

    # 3. Check train_multi_dataset.py
    try:
        multi = load_module_from_path("train_multi_dataset", os.path.join(project_root, "backend", "train_multi_dataset.py"))
        if "baseline_model" not in multi.STARTING_MODEL_PATH:
            print("  [FAIL] train_multi_dataset.py LOAD path is NOT baseline_model")
            all_passed = False
        else:
            print("  [OK] train_multi_dataset.py correctly points to baseline_model.")
    except Exception as e:
        print(f"  [FAIL] train_multi_dataset.py crashed: {e}")
        all_passed = False

    if not all_passed:
        print(f"\n>>> CRITICAL FAILURE at Loop {i}! <<<")
        sys.exit(1)

print("\n=======================================================")
print("SUCCESS: 10/10 VALIDATIONS PASSED. ZERO REGRESSIONS DETECTED.")
print("=======================================================")
