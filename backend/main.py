"""
NextChord SoloTab — FastAPI Backend
====================================
アコースティックギターインスト解析サーバー。
ポート8001で起動 (NextChordの8000と共存)。
"""
# pyre-ignore-all-errors
# pyright: reportMissingImports=false, reportCallIssue=false
# type: ignore
# flake8: noqa

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import sys
import json
import shutil
import uuid
import subprocess
import datetime as dt
import time
from typing import Optional, List
from pathlib import Path
from enum import Enum
import numpy as np

# solotab_utils import で NumPy/collections/ffmpeg パッチが自動適用
from solotab_utils import TUNINGS  # noqa: F401

# BasicPitch/TF等のWARNING:root:ログを抑制（uvicornのログは保持）
import logging
logging.getLogger().setLevel(logging.ERROR)
# uvicornのロガーは独立なので影響なし

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# venv312 の Python
VENV_DIR = PROJECT_ROOT.parent / "nextchord" / "venv312"
PYTHON_PATH = str(VENV_DIR / "Scripts" / "python.exe")

# FFMPEG_PATH / YT_DLP_PATH（エンドポイントで参照）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT.parent / "nextchord" / ".env")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
YT_DLP_PATH = os.getenv("YT_DLP_PATH", "yt-dlp")
if not shutil.which(YT_DLP_PATH):
    venv_yt = VENV_DIR / "Scripts" / "yt-dlp.exe"
    if venv_yt.exists():
        YT_DLP_PATH = str(venv_yt)
print(f"[SoloTab] FFMPEG: {shutil.which('ffmpeg') or FFMPEG_PATH}, yt-dlp: {YT_DLP_PATH}")


# Uploads
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Models (lazy load) ---
# Removed global model definitions here to speed up startup time.
# Models will be imported and loaded on-demand in the pipeline tasks.

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_sessions()
    # バックグラウンドでMoEモデルをプリロード（サーバー応答をブロックしない）
    import threading
    def _preload_models():
        try:
            t0 = time.time()
            from pure_moe_transcriber import _FULL_STAGES, _DOMAINS, _CACHED_MODELS
            import torch
            mt_dir = os.path.join(os.path.dirname(__file__), "..", "music-transcription", "python")
            if mt_dir not in sys.path:
                sys.path.insert(0, mt_dir)
            from model import architecture  # type: ignore
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded = 0
            for dname in _DOMAINS:
                for suffix in _FULL_STAGES:
                    key = f"finetuned_{dname}_{suffix}"
                    path = os.path.join(mt_dir, "_processed_guitarset_data", "training_output", key, "best_model.pth")
                    if os.path.exists(path) and key not in _CACHED_MODELS:
                        model = architecture.GuitarTabCRNN(
                            num_frames_rnn_input_dim=1280, rnn_type="GRU",
                            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
                        )
                        sd = torch.load(path, map_location=device, weights_only=False)
                        if list(sd.keys())[0].startswith("module."):
                            sd = {k[7:]: v for k, v in sd.items()}
                        model.load_state_dict(sd)
                        model.to(device)
                        model.eval()
                        _CACHED_MODELS[key] = model
                        loaded += 1
            print(f"[Preload] {loaded} MoE models cached on {device} in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[Preload] Failed (non-blocking): {e}", flush=True)
            import traceback; traceback.print_exc()
    threading.Thread(target=_preload_models, daemon=True).start()
    yield

app = FastAPI(
    title="NextChord SoloTab API",
    description="アコースティックギターインスト解析API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Session Management ---
class SessionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

sessions: dict = {}

def save_session(session_id: str):
    if session_id in sessions:
        session_dir = Path(sessions[session_id]["session_dir"])
        with open(session_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(sessions[session_id], f, ensure_ascii=False, indent=2)

SESSION_MAX_COUNT = 20

def load_all_sessions():
    global sessions
    if not UPLOAD_DIR.exists():
        return
    all_sessions: List[tuple] = []
    for s_dir in UPLOAD_DIR.iterdir():
        if not s_dir.is_dir():
            continue
        s_file = s_dir / "session.json"
        if not s_file.exists():
            continue
        try:
            with open(s_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["session_dir"] = str(s_dir)
            # Reset pending/processing → failed
            if data.get("status") in [SessionStatus.PENDING, SessionStatus.PROCESSING]:
                data["status"] = SessionStatus.FAILED
                data["error"] = "サーバー再起動により中断"
            all_sessions.append((s_dir.name, data))
        except Exception:
            continue

    all_sessions.sort(key=lambda x: x[0], reverse=True)
    for i, item in enumerate(all_sessions):
        if i >= SESSION_MAX_COUNT:
            break
        sid, data = item
        sessions[sid] = data

    print(f"[SoloTab] Loaded {min(len(all_sessions), SESSION_MAX_COUNT)} sessions")


# --- Request/Response Models ---
class YouTubeRequest(BaseModel):
    url: str
    tuning: str = "standard"

class UploadResponse(BaseModel):
    session_id: str
    message: str
    status: SessionStatus
    audio_url: Optional[str] = None

class StatusResponse(BaseModel):
    session_id: str
    status: SessionStatus
    progress: Optional[str] = None
    error: Optional[str] = None
    filename: Optional[str] = None

class ResultResponse(BaseModel):
    session_id: str
    status: SessionStatus
    bpm: Optional[float] = None
    filename: Optional[str] = None
    total_notes: Optional[int] = None
    tuning: Optional[str] = None
    key: Optional[str] = None
    capo: Optional[int] = None
    suggested_tuning: Optional[str] = None
    noise_gate: Optional[float] = None


# --- Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...),
                       tuning: str = Form("standard"),
                       skip_demucs: bool = Form(False),
                       fast_moe: bool = Form(True),
                       background_tasks: BackgroundTasks = None):
    """音声ファイルをアップロードして解析開始"""
    session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    audio_path = session_dir / file.filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Convert to WAV if needed
    wav_path = session_dir / "converted.wav"
    if audio_path.suffix.lower() != ".wav":
        try:
            subprocess.run(
                [FFMPEG_PATH, "-y", "-i", str(audio_path), "-ar", "22050", "-ac", "1", str(wav_path)],
                check=True, capture_output=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
    else:
        shutil.copy2(str(audio_path), str(wav_path))

    # Create session
    sessions[session_id] = {
        "session_dir": str(session_dir),
        "filename": file.filename,
        "wav_path": str(wav_path),
        "status": SessionStatus.PENDING,
        "progress": "アップロード完了",
        "error": None,
        "tuning": tuning if tuning in TUNINGS else "standard",
        "skip_demucs": skip_demucs,
        "fast_moe": fast_moe,
    }
    save_session(session_id)

    # Start pipeline in background
    background_tasks.add_task(_run_pipeline_bg, session_id)

    return UploadResponse(
        session_id=session_id,
        message="解析を開始しました",
        status=SessionStatus.PENDING,
        audio_url=f"/files/{session_id}/converted.wav"
    )


# --- YouTube Download ---

def download_youtube_audio(url: str, output_dir: Path) -> tuple:
    """Download audio from YouTube using yt-dlp. Returns (audio_path, metadata_dict)."""
    meta = {"title": "YouTube Video", "artist": ""}
    try:
        info_cmd = [
            YT_DLP_PATH, "--no-playlist", "--no-warnings",
            "--print", "%(title)s\n%(artist,uploader)s",
            url
        ]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=15)
        if info_result.returncode == 0 and info_result.stdout.strip():
            lines = info_result.stdout.strip().split("\n")
            if len(lines) >= 1 and lines[0].strip():
                meta["title"] = lines[0].strip()
            if len(lines) >= 2 and lines[1].strip() and lines[1].strip() != "NA":
                meta["artist"] = lines[1].strip()
            print(f"[YouTube] Title: {meta['title']}, Artist: {meta['artist']}")
    except Exception as e:
        print(f"[YouTube] Could not get metadata: {e}")

    temp_name = "download_temp"
    output_path = output_dir / temp_name

    cmd = [
        YT_DLP_PATH,
        "--no-playlist",
        "--no-warnings",
        "--ffmpeg-location", FFMPEG_BIN_DIR,
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path) + ".%(ext)s",
        url
    ]

    print(f"[SoloTab] Downloading YouTube audio: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr}")
        raise Exception(f"YouTube download failed: {result.stderr}")

    time.sleep(1)

    wav_path = output_dir / f"{temp_name}.wav"
    if wav_path.exists():
        return wav_path, meta

    for f in output_dir.glob(f"{temp_name}.*"):
        if f.suffix.lower() in [".mp3", ".m4a", ".webm", ".opus", ".wav"]:
            return f, meta

    raise FileNotFoundError("Could not find downloaded YouTube audio file.")


@app.post("/upload/youtube", response_model=UploadResponse)
async def upload_youtube(background_tasks: BackgroundTasks, request: YouTubeRequest):
    """YouTube URLを受け取って解析を開始"""
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + "yt-" + str(uuid.uuid4().hex)[:6]  # type: ignore
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    sessions[session_id] = {
        "session_dir": str(session_dir),
        "filename": "YouTube Video",
        "url": url,
        "status": SessionStatus.PENDING,
        "progress": "YouTube音声をダウンロード中...",
        "error": None,
        "tuning": request.tuning if request.tuning in TUNINGS else "standard",
    }
    save_session(session_id)

    def process_youtube():
        try:
            audio_path, yt_meta = download_youtube_audio(url, session_dir)
            sessions[session_id]["filename"] = yt_meta["title"]
            save_session(session_id)

            # WAV変換
            final_wav = session_dir / "converted.wav"
            if audio_path.suffix.lower() != ".wav":
                subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", str(audio_path), "-ar", "22050", "-ac", "1", str(final_wav)],
                    check=True, capture_output=True
                )
                audio_path.unlink(missing_ok=True)
            else:
                if audio_path != final_wav:
                    shutil.move(str(audio_path), str(final_wav))

            sessions[session_id]["wav_path"] = str(final_wav)
            sessions[session_id]["progress"] = "ダウンロード完了。解析を開始..."
            save_session(session_id)

            # パイプライン実行
            _run_pipeline_bg(session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sessions[session_id]["status"] = SessionStatus.FAILED
            sessions[session_id]["error"] = f"YouTube解析エラー: {str(e)}"
            sessions[session_id]["progress"] = "エラー"
            save_session(session_id)

    background_tasks.add_task(process_youtube)

    return UploadResponse(
        session_id=session_id,
        message="YouTubeダウンロードと解析を開始しました",
        status=SessionStatus.PENDING,
    )

def _run_pipeline_bg(session_id: str):
    """Background task: run the analysis pipeline."""
    import importlib
    import pipeline as _pipeline_mod
    importlib.reload(_pipeline_mod)
    run_pipeline = _pipeline_mod.run_pipeline

    session = sessions[session_id]
    session_dir = Path(session["session_dir"])
    wav_path = Path(session["wav_path"])
    tuning_name = session.get("tuning", "standard")

    # パイプライン内部ステップ → フロントエンド5ステップのマッピング
    STEP_MAP = {
        "beats": 1, "key": 1, "capo": 1,
        "demucs": 2, "preprocess": 2,
        "notes": 2, "spectral": 2,
        "filter": 3, "assign": 3, "note_filter": 3, "quantize": 3,
        "technique": 3, "technique_pm": 3, "tuning_detect": 3, "chords": 3,
        "musicxml": 4,
    }

    def progress_cb(step: str, msg: str):
        session["progress"] = msg
        session["_current_step"] = step
        # steps_done = 完了済みフロントエンドステップ数
        mapped = STEP_MAP.get(step, session.get("steps_done", 0))
        session["steps_done"] = mapped

    try:
        session["status"] = SessionStatus.PROCESSING
        save_session(session_id)

        # タイトル: filename から音声ファイル拡張子のみ除去
        song_title = session.get("filename", session_id)
        audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
        if song_title:
            import os
            _, ext = os.path.splitext(song_title)
            if ext.lower() in audio_exts:
                song_title = song_title[:-len(ext)]

        result = run_pipeline(
            session_id, session_dir, wav_path,
            tuning_name=tuning_name,
            title=song_title,
            progress_cb=progress_cb,
            skip_demucs=session.get("skip_demucs", False),
            fast_moe=session.get("fast_moe", True),
        )

        session["status"] = SessionStatus.COMPLETED
        session["bpm"] = result["bpm"]
        session["time_signature"] = result.get("time_signature", "4/4")
        session["total_notes"] = result["total_notes"]
        session["key"] = result.get("key")
        session["capo"] = result.get("capo", 0)
        session["suggested_tuning"] = result.get("suggested_tuning")
        session["result"] = result
        session["progress"] = "解析完了"
        session["steps_done"] = 4  # 全ステップ完了
        save_session(session_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        session["status"] = SessionStatus.FAILED
        session["error"] = str(e)
        session["progress"] = "エラー"
        save_session(session_id)


@app.get("/status/{session_id}/stream")
async def stream_status(session_id: str):
    """SSE (Server-Sent Events) で進捗配信"""
    import asyncio
    from starlette.responses import StreamingResponse

    async def event_generator():
        last_progress = None
        while True:
            if session_id not in sessions:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                return

            session = sessions[session_id]
            steps_done = session.get("steps_done", 0)
            current = {
                "status": session.get("status", "pending"),
                "progress": session.get("progress", ""),
                "filename": session.get("filename"),
                "steps_done": steps_done,
            }

            progress_key = f"{current['status']}:{current['progress']}:{steps_done}"
            if progress_key != last_progress:
                yield f"data: {json.dumps(current, ensure_ascii=False)}\n\n"
                last_progress = progress_key

            if current["status"] in ("completed", "failed"):
                if current["status"] == "failed":
                    current["error"] = session.get("error", "")
                    yield f"data: {json.dumps(current, ensure_ascii=False)}\n\n"
                return

            await asyncio.sleep(0.8)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    return StatusResponse(
        session_id=session_id,
        status=s["status"],
        progress=s.get("progress"),
        error=s.get("error"),
        filename=s.get("filename"),
    )


@app.get("/result/{session_id}", response_model=ResultResponse)
async def get_result(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    if s["status"] != SessionStatus.COMPLETED:
        raise HTTPException(status_code=202, detail="Analysis not complete")

    return ResultResponse(
        session_id=session_id,
        status=s["status"],
        bpm=s.get("bpm"),
        filename=s.get("filename"),
        total_notes=s.get("total_notes"),
        tuning=s.get("tuning"),
        key=s.get("key"),
        capo=s.get("capo"),
        suggested_tuning=s.get("suggested_tuning"),
        noise_gate=s.get("noise_gate"),
    )


@app.get("/result/{session_id}/musicxml")
async def get_musicxml(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    xml_path = session_dir / "tab.musicxml"
    if not xml_path.exists():
        raise HTTPException(status_code=404, detail="MusicXML not generated")
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    filename = s.get("filename", session_id)
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]
    from urllib.parse import quote
    safe_filename = f"{filename}.musicxml"
    try:
        safe_filename.encode("latin-1")
        cd = f'attachment; filename="{safe_filename}"'
    except UnicodeEncodeError:
        cd = f"attachment; filename*=UTF-8''{quote(safe_filename)}"
    return Response(
        content=content,
        media_type="application/xml",
        headers={"Content-Disposition": cd},
    )


@app.get("/result/{session_id}/gp5")
async def get_gp5(session_id: str):
    """GP5ファイルを返す（AlphaTab表示用 + TuxGuitarダウンロード用）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    gp5_path = session_dir / "tab.gp5"
    if not gp5_path.exists():
        # 古いセッション: notes_assigned.jsonからGP5を自動生成
        assigned_path = session_dir / "notes_assigned.json"
        if assigned_path.exists():
            try:
                with open(assigned_path, "r", encoding="utf-8") as f:
                    notes = json.load(f)
                if isinstance(notes, dict):
                    notes = notes.get("notes", notes)
                _regenerate_musicxml(session_id, notes)
            except Exception as e:
                print(f"[get_gp5] Auto-generation failed: {e}")
    if not gp5_path.exists():
        raise HTTPException(status_code=404, detail="GP5 not generated")

    filename = s.get("filename", session_id)
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]
    from urllib.parse import quote
    safe_filename = f"{filename}.gp5"
    try:
        safe_filename.encode("latin-1")
        cd = f'attachment; filename="{safe_filename}"'
    except UnicodeEncodeError:
        cd = f"attachment; filename*=UTF-8''{quote(safe_filename)}"
    return FileResponse(
        str(gp5_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": cd},
    )


@app.get("/result/{session_id}/gp4")
async def get_gp4(session_id: str):
    """GP4ファイルを返す（TuxGuitar互換用）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    gp5_path = session_dir / "tab.gp5"
    gp4_path = session_dir / "tab.gp4"

    if not gp4_path.exists() and gp5_path.exists():
        try:
            import guitarpro as gp
            song = gp.parse(str(gp5_path))
            gp.write(song, str(gp4_path))
            print(f"[get_gp4] Converted GP5 -> GP4: {gp4_path}")
        except Exception as e:
            print(f"[get_gp4] Conversion failed: {e}")
            import traceback; traceback.print_exc()

    if not gp4_path.exists():
        raise HTTPException(status_code=404, detail="GP4 not generated")

    filename = s.get("filename", session_id)
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]
    from urllib.parse import quote
    safe_filename = f"{filename}.gp4"
    try:
        safe_filename.encode("latin-1")
        cd = f'attachment; filename="{safe_filename}"'
    except UnicodeEncodeError:
        cd = f"attachment; filename*=UTF-8''{quote(safe_filename)}"
    return FileResponse(
        str(gp4_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": cd},
    )


@app.post("/result/{session_id}/open-tuxguitar")
async def open_tuxguitar(session_id: str):
    """GP4を生成してTuxGuitar（OSのデフォルトアプリ）で開く"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    gp5_path = session_dir / "tab.gp5"
    gp4_path = session_dir / "tab.gp4"

    # GP4がなければGP5から変換
    if not gp4_path.exists() and gp5_path.exists():
        try:
            import guitarpro as gp
            song = gp.parse(str(gp5_path))
            gp.write(song, str(gp4_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GP4変換失敗: {e}")

    if not gp4_path.exists():
        raise HTTPException(status_code=404, detail="GP4ファイルが見つかりません")

    # OSのデフォルトアプリで開く (Windows: os.startfile)
    import os, platform
    try:
        abs_path = str(gp4_path.resolve())
        if platform.system() == "Windows":
            os.startfile(abs_path)
        elif platform.system() == "Darwin":
            import subprocess
            subprocess.Popen(["open", abs_path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", abs_path])
        print(f"[open-tuxguitar] Opened: {abs_path}")
        return {"status": "ok", "path": abs_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"起動失敗: {e}")


@app.get("/result/{session_id}/pdf")
async def get_pdf(session_id: str):
    """MusicXMLからTAB譜PDFを生成してダウンロードさせる"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    xml_path = session_dir / "tab.musicxml"
    if not xml_path.exists():
        raise HTTPException(status_code=404, detail="MusicXML not generated")

    pdf_path = session_dir / "tab.pdf"

    # 常にreportlab TABレンダラーで再生成（MuseScore版はTABスタッフが欠落するため）
    xml_path = session_dir / "tab.musicxml"
    if xml_path.exists():
        try:
            from pdf_renderer import musicxml_to_pdf
            musicxml_to_pdf(str(xml_path), str(pdf_path), title=s.get("filename", "Guitar TAB"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="PDF file was not created")

    filename = s.get("filename", session_id)
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]

    from urllib.parse import quote
    safe_filename = f"{filename}.pdf"
    try:
        safe_filename.encode("latin-1")
        cd = f'attachment; filename="{safe_filename}"'
    except UnicodeEncodeError:
        cd = f"attachment; filename*=UTF-8''{quote(safe_filename)}"

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": cd},
    )


def _regenerate_musicxml(session_id: str, notes: list,
                         tuning: list = None, noise_gate: float = None):
    """notes → tab.gp5 + tab.musicxml 再生成の共通関数"""
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])

    if tuning is None:
        tuning_name = s.get("tuning", "standard")
        tuning = TUNINGS.get(tuning_name, TUNINGS["standard"])
        capo = s.get("capo", 0)
        if capo and capo > 0:
            tuning = [p + capo for p in tuning]

    beats, bpm = [], s.get("bpm", 120)
    time_sig = s.get("time_signature", "4/4")
    rhythm_info = None
    beats_path = session_dir / "beats.json"
    if beats_path.exists():
        with open(beats_path, "r") as f:
            bd = json.load(f)
        beats = bd.get("beats", [])
        bpm = bd.get("bpm", bpm)
        time_sig = bd.get("time_signature", time_sig)

    title_raw = s.get("filename", session_id)
    # Remove audio extension first
    import re as _re
    for ext in ('.mp3','.wav','.m4a','.flac','.ogg','.opus','.webm','.mp4'):
        if title_raw.lower().endswith(ext):
            title_raw = title_raw[:-len(ext)]
            break
    # Clean: remove common junk metadata patterns from filename
    # e.g. "(128k)", "ギター Tab譜 楽譜", "コードネーム付", etc.
    _junk_patterns = [
        r'\s*\(\d+k\)',                     # (128k)
        r'\s*Tab譜.*$',                       # Tab譜 楽譜 ... trailing
        r'\s*ギター\s*タブ.*$',               # ギター タブ ...
        r'\s*コードネーム付\s*',              # コードネーム付
        r'\s*-\s*アコースティック.*$',        # - アコースティック ...
        r'\s*楽譜.*$',                        # 楽譜...
    ]
    title_clean = title_raw.strip()
    for pat in _junk_patterns:
        title_clean = _re.sub(pat, '', title_clean, flags=_re.IGNORECASE).strip()
    if not title_clean:
        title_clean = title_raw.strip()
    # GP5 binary format uses Latin-1 encoding internally
    try:
        title_clean.encode('latin-1')
        title = title_clean
    except (UnicodeEncodeError, UnicodeDecodeError):
        title = _re.sub(r'[^\x20-\x7E]', '', title_clean).strip() or session_id
    gate = noise_gate if noise_gate is not None else 0.20

    # --- GP5再生成 ---
    try:
        from gp_renderer import notes_to_gp5
        gp5_bytes = notes_to_gp5(
            notes, beats=beats, bpm=bpm, title=title,
            tuning=tuning, time_signature=time_sig,
            rhythm_info=rhythm_info, noise_gate=gate,
        )
        with open(session_dir / "tab.gp5", "wb") as f:
            f.write(gp5_bytes)
        # GP4 (TuxGuitar用) も同時生成
        try:
            import guitarpro as _gp
            _song = _gp.parse(str(session_dir / "tab.gp5"))
            _gp.write(_song, str(session_dir / "tab.gp4"))
        except Exception:
            pass
    except Exception as e:
        print(f"[_regenerate] GP5 generation failed: {e}")

    # --- MusicXML再生成 ---
    from tab_renderer import notes_to_tab_musicxml
    kwargs = dict(
        beats=beats, bpm=bpm,
        title=title,
        tuning=tuning,
        time_signature=time_sig,
    )
    if noise_gate is not None:
        kwargs["noise_gate"] = noise_gate
    xml_content, tech_map = notes_to_tab_musicxml(notes, **kwargs)

    with open(session_dir / "tab.musicxml", "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Delete stale PDF so it gets regenerated on next request
    pdf_path = session_dir / "tab.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    return xml_content, tech_map


class RetuneRequest(BaseModel):
    tuning: str
    capo: Optional[int] = 0
    noise_gate: Optional[float] = 0.30


@app.post("/result/{session_id}/retune")
async def retune(session_id: str, request: RetuneRequest):
    """チューニングを変更して弦/フレット再割り当て + MusicXML再生成（ノート検出はスキップ）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    tuning_name = request.tuning
    if tuning_name not in TUNINGS:
        raise HTTPException(status_code=400, detail=f"Unknown tuning: {tuning_name}. Available: {list(TUNINGS.keys())}")

    s = sessions[session_id]
    session_dir = Path(s["session_dir"])

    import copy

    # オリジナルノートを保持: retune後も初期状態に戻れるようにする
    original_path = session_dir / "notes_assigned_original.json"
    assigned_path = session_dir / "notes_assigned.json"

    # 初回retune時にオリジナルをバックアップ
    if not original_path.exists() and assigned_path.exists():
        shutil.copy2(assigned_path, original_path)

    # 常にオリジナルから読み込む（retune結果で上書きされない）
    notes_path = original_path if original_path.exists() else assigned_path
    if not notes_path.exists():
        notes_path = session_dir / "notes.json"
    if not notes_path.exists():
        raise HTTPException(status_code=404, detail="Notes data not found. Run analysis first.")

    with open(notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)

    # notes_assigned.jsonはリスト直接、notes.jsonは{"notes": [...]}
    notes = copy.deepcopy(notes_data if isinstance(notes_data, list) else notes_data.get("notes", notes_data))
    tuning = TUNINGS[tuning_name]

    # カポ対応: tuningにカポ分を加算
    capo = request.capo if request.capo is not None else s.get("capo", 0)
    if capo is None:
        capo = 0
    capo_tuning = [p + capo for p in tuning] if capo > 0 else tuning

    # --- 共鳴音フィルタ (retune時にも適用) ---
    # MoE出力分析結果:
    # - 各拍は正確に3ノート (3連符アルペジオ)
    # - パターン: [G3(55), melody, accomp(59)] が頻出
    # - 正解: [melody, accomp, melody] の交互
    # - G3は3弦開放の共鳴音で、melodyのタイムスロットに入り込んでいる
    # 修正: 各拍の1音目がG3(55)で、同拍内に別のmelody音がある場合、
    #       G3をその拍のmelody音で置換する
    sympa_removed = 0
    beats_path = session_dir / "beats.json"
    if beats_path.exists() and len(notes) > 10:
        import numpy as np
        beats_data = json.load(open(beats_path, "r", encoding="utf-8"))
        beats = beats_data if isinstance(beats_data, list) else beats_data.get("beats", [])
        if len(beats) > 2:
            beats_arr = np.array(beats)
            
            # 各ノートをビートに割り当て
            note_beat_idx = []
            for n in notes:
                t = float(n.get('start', 0))
                bi = int(np.searchsorted(beats_arr, t, side='right')) - 1
                bi = max(0, min(bi, len(beats_arr) - 1))
                note_beat_idx.append(bi)
            
            # ビートごとにノートをグループ化
            from collections import defaultdict
            beat_groups = defaultdict(list)
            for ni, bi in enumerate(note_beat_idx):
                beat_groups[bi].append(ni)
            
            # 各拍のmelody pitch(最高音 - bassを除く)を収集
            beat_melody = {}
            BASS_RANGE = {40, 43, 45, 47, 48, 50}  # E2-D3
            for bi, indices in sorted(beat_groups.items()):
                pitches = [int(notes[i].get('pitch', 0)) for i in indices]
                non_bass = [p for p in pitches if p not in BASS_RANGE and p != 55]
                if non_bass:
                    beat_melody[bi] = max(non_bass)  # 最高音 = melody
            
            # G3(55)をmelody音で置換
            remove_indices = set()
            for bi, indices in sorted(beat_groups.items()):
                g3_indices = [i for i in indices if int(notes[i].get('pitch', 0)) == 55]
                if not g3_indices:
                    continue
                
                # この拍のmelody音を取得(G3以外の最高音)
                non_g3 = [int(notes[i].get('pitch', 0)) for i in indices 
                          if int(notes[i].get('pitch', 0)) != 55 
                          and int(notes[i].get('pitch', 0)) not in BASS_RANGE]
                
                if non_g3:
                    # G3→melody pitchに単純置換
                    # mel-acc-mel強制は非G3拍やbass拍を壊すため不採用
                    melody_pitch = max(non_g3)
                    for gi in g3_indices:
                        notes[gi]['pitch'] = melody_pitch
                        sympa_removed += 1
                else:
                    # この拍にmelody音がない → G3がmelodyの代わり
                    # 前後の拍のmelodyを参照
                    prev_mel = beat_melody.get(bi - 1)
                    next_mel = beat_melody.get(bi + 1)
                    replacement = prev_mel or next_mel
                    if replacement:
                        for gi in g3_indices:
                            notes[gi]['pitch'] = replacement
                            sympa_removed += 1
                            print(f"[retune] G3→pitch={replacement} at t={notes[gi]['start']:.3f}")
            
            if remove_indices:
                removed_count = len(remove_indices)
                notes = [n for i, n in enumerate(notes) if i not in remove_indices]
                sympa_removed += removed_count
            
            if sympa_removed > 0:
                print(f"[retune] 共鳴音フィルタ: {sympa_removed}ノート修正/除去")


    # Re-run string assignment
    from string_assigner import assign_strings_dp
    notes = assign_strings_dp(notes, tuning=capo_tuning)

    # フレットクランプ: パイプラインと同等の上限制約
    MAX_FRET = 12
    for n in notes:
        if n.get("fret", 0) > MAX_FRET:
            pitch = n.get("pitch", 60)
            base_tuning = TUNINGS.get(tuning_name, TUNINGS["standard"])
            best_str, best_fret = None, 99
            for s_idx, open_pitch in enumerate(base_tuning):
                s_num = 6 - s_idx
                f = pitch - open_pitch
                if 0 <= f <= MAX_FRET and (best_str is None or f < best_fret):
                    best_str, best_fret = s_num, f
            if best_str is not None:
                n["string"] = best_str
                n["fret"] = best_fret

    # 左手指番号割り当て
    try:
        from finger_assigner import assign_fingers
        notes = assign_fingers(notes)
    except Exception as e:
        print(f"[retune] 指番号割り当てスキップ: {e}")

    # Save reassigned notes（オリジナルは保持、表示用のみ上書き）
    with open(assigned_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    # Re-generate MusicXML
    _regenerate_musicxml(session_id, notes, tuning=capo_tuning, noise_gate=request.noise_gate)

    # techniques.jsonはカポ/チューニングに依存しない → オリジナルを保持
    tech_original = session_dir / "techniques_original.json"
    tech_current = session_dir / "techniques.json"
    if not tech_original.exists() and tech_current.exists():
        shutil.copy2(tech_current, tech_original)
    if tech_original.exists():
        shutil.copy2(tech_original, tech_current)

    # Update session
    s["tuning"] = tuning_name
    s["capo"] = capo
    gate = request.noise_gate if request.noise_gate is not None else 0.20
    from gp_renderer import _filter_noise
    filtered_count = len(_filter_noise(notes, gate))
    s["total_notes"] = filtered_count
    if request.noise_gate is not None:
        s["noise_gate"] = request.noise_gate
    save_session(session_id)

    return {"status": "ok", "tuning": tuning_name, "capo": capo, "total_notes": filtered_count}


class NoteEditRequest(BaseModel):
    fret: Optional[int] = None
    string: Optional[int] = None
    delete: Optional[bool] = False


@app.patch("/result/{session_id}/notes/{note_index}")
async def edit_note(session_id: str, note_index: int, request: NoteEditRequest):
    """ノートを編集（フレット/弦変更 or 削除）→ MusicXML再生成"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    assigned_path = session_dir / "notes_assigned.json"
    if not assigned_path.exists():
        raise HTTPException(status_code=404, detail="Notes not found")
    
    with open(assigned_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    if note_index < 0 or note_index >= len(notes):
        raise HTTPException(status_code=400, detail=f"Invalid note index: {note_index}")

    if request.delete:
        notes.pop(note_index)
        action = "deleted"
    else:
        note = notes[note_index]
        if request.fret is not None:
            note["fret"] = request.fret
        if request.string is not None:
            note["string"] = request.string
        action = f"edited fret={note.get('fret')} string={note.get('string')}"

    with open(assigned_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    try:
        _regenerate_musicxml(session_id, notes)
    except Exception as e:
        print(f"[edit_note] Regeneration failed: {e}")
        import traceback; traceback.print_exc()

    s["total_notes"] = len(notes)
    save_session(session_id)

    return {"status": "ok", "action": action, "total_notes": len(notes)}


class NoteAddRequest(BaseModel):
    start: float          # 開始時間（秒）
    end: float            # 終了時間（秒）
    pitch: int            # MIDIノート番号
    string: int = 1       # 弦番号 (1-6)
    fret: int = 0         # フレット番号


@app.post("/result/{session_id}/notes")
async def add_note(session_id: str, request: NoteAddRequest):
    """ノートを追加 → MusicXML再生成"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    assigned_path = session_dir / "notes_assigned.json"
    if not assigned_path.exists():
        raise HTTPException(status_code=404, detail="Notes not found")
    
    with open(assigned_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    new_note = {
        "start": request.start,
        "end": request.end,
        "pitch": request.pitch,
        "string": request.string,
        "fret": request.fret,
        "velocity": 0.7,
        "technique": None,
    }

    # 時間順でソートされた位置に挿入
    insert_idx = 0
    for i, n in enumerate(notes):
        if float(n.get("start", 0)) > request.start:
            insert_idx = i
            break
        insert_idx = i + 1
    notes.insert(insert_idx, new_note)

    with open(assigned_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    _regenerate_musicxml(session_id, notes)

    s["total_notes"] = len(notes)
    save_session(session_id)

    return {"status": "ok", "action": "added", "note_index": insert_idx, "total_notes": len(notes)}


@app.get("/result/{session_id}/notes")
async def get_notes(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    notes_path = session_dir / "notes_assigned.json"
    if not notes_path.exists():
        return {"notes": []}
    with open(notes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"notes": data}


@app.get("/result/{session_id}/techniques")
async def get_techniques(session_id: str):
    """テクニックマップを返す（MusicXMLのノート順と対応）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    tech_path = session_dir / "techniques.json"
    if not tech_path.exists():
        return []
    with open(tech_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/result/{session_id}/beats")
async def get_beats(session_id: str):
    """ビートデータを返す（カーソル同期用）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    beats_path = session_dir / "beats.json"
    if not beats_path.exists():
        return {"beats": [], "bpm": 120}
    with open(beats_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.api_route("/files/{session_id}/{filename}", methods=["GET", "HEAD"])
async def get_file(session_id: str, filename: str):
    # sessions辞書から探す
    if session_id in sessions:
        session_dir = Path(sessions[session_id]["session_dir"])
    else:
        # フォールバック: ディスクから直接探す (リロード後など)
        session_dir = UPLOAD_DIR / session_id
    file_path = session_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/sessions")
async def get_sessions():
    history = []
    for sid in sorted(sessions.keys(), reverse=True):
        s = sessions[sid]
        history.append({
            "session_id": sid,
            "filename": s.get("filename", "Unknown"),
            "status": s.get("status"),
            "bpm": s.get("bpm"),
            "total_notes": s.get("total_notes"),
        })
    return history


@app.get("/health")
async def health():
    return {"status": "healthy", "app": "NextChord SoloTab", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    import socket

    # SO_REUSEADDR: TIME_WAITによるポート競合を防止
    # Windows環境でサーバー再起動時に「Address already in use」を回避
    _orig_bind = socket.socket.bind
    def _reuse_bind(self, address):
        self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return _orig_bind(self, address)
    socket.socket.bind = _reuse_bind

    uvicorn.run(app, host="0.0.0.0", port=8001)
