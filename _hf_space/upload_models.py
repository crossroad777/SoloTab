"""
SoloTab モデルを HF Models リポジトリにアップロードするスクリプト
"""
import os
from huggingface_hub import HfApi, create_repo

# === 設定 ===
# ここを自分の HF ユーザー名に変更してください
HF_USERNAME = "Crossroad777"
REPO_NAME = "solotab-moe-models"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

MODEL_BASE_DIR = r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output"

EXPERT_DIRS = [
    "finetuned_martin_finger_guitarset_ft",
    "finetuned_taylor_finger_guitarset_ft",
    "finetuned_luthier_finger_guitarset_ft",
    "finetuned_martin_pick_guitarset_ft",
    "finetuned_taylor_pick_guitarset_ft",
    "finetuned_luthier_pick_guitarset_ft",
]

def main():
    api = HfApi()

    # リポジトリ作成（存在しなければ）
    print(f"Creating/verifying repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # 各モデルをアップロード
    for expert_dir in EXPERT_DIRS:
        model_path = os.path.join(MODEL_BASE_DIR, expert_dir, "best_model.pth")
        if not os.path.exists(model_path):
            print(f"SKIP (not found): {model_path}")
            continue

        remote_path = f"{expert_dir}/best_model.pth"
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Uploading {expert_dir} ({size_mb:.1f} MB) -> {remote_path}")

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  -> Done.")

    print(f"\nAll models uploaded to: https://huggingface.co/{REPO_ID}")
    print(f"\napp.py の HF_MODEL_REPO を '{REPO_ID}' に更新してください。")


if __name__ == "__main__":
    main()
