"""
SoloTab HF Space をデプロイするスクリプト
"""
import os
from huggingface_hub import HfApi, create_repo

# === 設定 ===
# ここを自分の HF ユーザー名に変更してください
HF_USERNAME = "Crossroad777"
SPACE_NAME = "solotab"
SPACE_ID = f"{HF_USERNAME}/{SPACE_NAME}"

SPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# アップロード対象ファイル
FILES_TO_UPLOAD = [
    "README.md",
    "app.py",
    "config.py",
    "requirements.txt",
    "model/__init__.py",
    "model/architecture.py",
]


def main():
    api = HfApi()

    # Space 作成
    print(f"Creating/verifying Space: {SPACE_ID}")
    create_repo(
        SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    # ファイルアップロード
    for rel_path in FILES_TO_UPLOAD:
        local_path = os.path.join(SPACE_DIR, rel_path)
        if not os.path.exists(local_path):
            print(f"SKIP (not found): {local_path}")
            continue

        print(f"Uploading {rel_path} ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=SPACE_ID,
            repo_type="space",
        )

    print(f"\nSpace deployed at: https://huggingface.co/spaces/{SPACE_ID}")
    print("ビルドが完了するまで数分お待ちください。")


if __name__ == "__main__":
    main()
