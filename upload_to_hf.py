import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

SPACE_ID = "Crossroad777/solotab"
STAGE_DIR = Path(r"D:\Music\nextchord-solotab\_hf_stage")

# Patch main.py to serve frontend static files and use port 7860
main_py = STAGE_DIR / "backend" / "main.py"
content = main_py.read_text(encoding="utf-8")

# Add static file serving at the end of main.py if not already present
if "StaticFiles" not in content:
    static_mount = '''

# === HuggingFace Spaces: Serve frontend static files ===
import os
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(FRONTEND_DIR):
    from starlette.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
'''
    content += static_mount
    main_py.write_text(content, encoding="utf-8")
    print("Patched main.py with static file serving")

# Also patch the API base URL in frontend if needed
vite_config = STAGE_DIR / "frontend" / "vite.config.js"
if vite_config.exists():
    vc = vite_config.read_text(encoding="utf-8")
    print(f"Vite config: {vc[:200]}")

# Patch frontend App.jsx to use relative API URL
app_jsx = STAGE_DIR / "frontend" / "src" / "App.jsx"
if app_jsx.exists():
    app_content = app_jsx.read_text(encoding="utf-8")
    # Change API_BASE from localhost to relative
    app_content = app_content.replace(
        '"http://localhost:8001"',
        '""'  # Empty string = relative URL (same origin)
    )
    app_jsx.write_text(app_content, encoding="utf-8")
    print("Patched App.jsx API_BASE to relative URL")

# Upload to HuggingFace
print(f"\nUploading {STAGE_DIR} to {SPACE_ID}...")
api = HfApi()
api.upload_folder(
    folder_path=str(STAGE_DIR),
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="Deploy SoloTab - full app with models",
)
print(f"\nDone! Visit: https://huggingface.co/spaces/{SPACE_ID}")
