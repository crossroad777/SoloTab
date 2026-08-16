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

# v2.2: 既知の無害な警告を抑制
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore", message=".*n_fft.*too large.*")  # librosa short segment
warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*charset_normalizer.*")  # requests
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")  # tensorflow
warnings.filterwarnings("ignore", message=".*Empty filters detected.*")  # librosa mel
warnings.filterwarnings("ignore", category=DeprecationWarning, module="madmom")  # numpy/madmom

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

# TensorFlow oneDNN警告を抑制
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# venv312 の Python
VENV_DIR = PROJECT_ROOT.parent / "nextchord" / "venv312"
PYTHON_PATH = str(VENV_DIR / "Scripts" / "python.exe")

# FFMPEG_PATH / YT_DLP_PATH（エンドポイントで参照）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT.parent / "nextchord" / ".env")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
# yt-dlp に渡すための ffmpeg ディレクトリを解決
_ffmpeg_dir = os.path.dirname(shutil.which(FFMPEG_PATH) or FFMPEG_PATH)
FFMPEG_BIN_DIR = _ffmpeg_dir if _ffmpeg_dir else None 
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
# All models are preloaded in a single background thread at startup.
# Each module has its own cache mechanism, so we just trigger the load.

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_sessions()
    yield

app = FastAPI(
    title="NextChord SoloTab API",
    description="アコースティックギターインスト解析API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://solotab.vercel.app",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://localhost:8001",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://.*\.trycloudflare\.com|http://192\.168\.\d+\.\d+:\d+|http://10\.\d+\.\d+\.\d+:\d+|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静的ファイル配信 ──────────────────────────────────────────────────
# gprotab_downloads: 本物のGP5ファイル群（検証・学習用）
# http://localhost:8001/gprotab/{filename}
from fastapi.staticfiles import StaticFiles

_gprotab_dir = PROJECT_ROOT / "gprotab_downloads"
if _gprotab_dir.exists():
    app.mount("/gprotab", StaticFiles(directory=str(_gprotab_dir)), name="gprotab")
    print(f"[SoloTab] GP files served: /gprotab/ -> {_gprotab_dir}")

# backend/uploads: セッションファイル（GP5/JSON）
_backend_uploads = PROJECT_ROOT / "backend" / "uploads"
if _backend_uploads.exists():
    app.mount("/backend-uploads", StaticFiles(directory=str(_backend_uploads)), name="backend-uploads")
    print(f"[SoloTab] Backend uploads served: /backend-uploads/ -> {_backend_uploads}")

# プロジェクトルート: test_techniques_verify.html等
app.mount("/verify", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="verify")
print(f"[SoloTab] Verify page served: /verify/ -> {PROJECT_ROOT}")



# --- Session Management ---
class SessionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

import threading
_SESSIONS_LOCK = threading.Lock()
sessions: dict = {}

def save_session(session_id: str):
    with _SESSIONS_LOCK:
        if session_id not in sessions:
            return
        session_data = dict(sessions[session_id])
    session_dir = Path(session_data["session_dir"])
    with open(session_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

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
    with _SESSIONS_LOCK:
        sessions.clear()
        for i, item in enumerate(all_sessions):
            if i >= SESSION_MAX_COUNT:
                # ディレクトリは残すがメモリにはロードしない
                continue
            sid, data = item
            sessions[sid] = data

    print(f"[SoloTab] Loaded {min(len(all_sessions), SESSION_MAX_COUNT)} sessions")


# --- Request/Response Models ---
class YouTubeRequest(BaseModel):
    url: str
    tuning: str = "standard"
    guitar_type: str = "auto"

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
    steps_done: int = 0

class ResultResponse(BaseModel):
    session_id: str
    status: SessionStatus
    bpm: Optional[float] = None
    time_signature: Optional[str] = None
    filename: Optional[str] = None
    total_notes: Optional[int] = None
    tuning: Optional[str] = None
    key: Optional[str] = None
    capo: Optional[int] = None
    suggested_tuning: Optional[str] = None
    anchors: Optional[dict] = None
    noise_gate: Optional[float] = None


# --- Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...),
                       tuning: str = Form("standard"),
                       skip_demucs: bool = Form(False),
                       fast_moe: bool = Form(True),
                       guitar_type: str = Form("auto"),
                       transcription_profile: str = Form("standard"),
                       enable_technique_gp5: bool = Form(True),
                       enable_technique_overlay: bool = Form(False),
                       enable_technique_fingers: bool = Form(False),
                       background_tasks: BackgroundTasks = None):
    """音声ファイルをアップロードして解析開始"""
    session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + str(uuid.uuid4().hex)[:6]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    audio_path = session_dir / file.filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    is_midi_file = audio_path.suffix.lower() in (".mid", ".midi")
    midi_path = None

    # Convert to WAV if needed
    wav_path = session_dir / "converted.wav"
    if is_midi_file:
        midi_path = audio_path
        # MIDIファイル用のダミー/合成WAV（再生用）を生成
        try:
            # 5秒以上の無音/サイン波ダミーWAVを生成 (ブラウザ同期再生用)
            import soundfile as sf
            import numpy as np
            sr = 22050
            dummy_sig = np.zeros(sr * 3, dtype=np.float32)
            sf.write(str(wav_path), dummy_sig, sr)
        except Exception:
            pass
    elif audio_path.suffix.lower() != ".wav":
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
    with _SESSIONS_LOCK:
        sessions[session_id] = {
            "session_dir": str(session_dir),
            "filename": file.filename,
            "wav_path": str(wav_path),
            "midi_path": str(midi_path) if midi_path else None,
            "status": SessionStatus.PENDING,
            "progress": "アップロード完了",
            "error": None,
            "tuning": tuning if tuning in TUNINGS else "standard",
            "skip_demucs": skip_demucs or is_midi_file,
            "fast_moe": fast_moe,
            "guitar_type": guitar_type if guitar_type in ("auto", "steel", "nylon") else "auto",
            "transcription_profile": transcription_profile if transcription_profile in ("standard", "classic", "arpeggio") else "standard",
            "enable_technique_gp5": enable_technique_gp5,
            "enable_technique_overlay": enable_technique_overlay,
            "enable_technique_fingers": enable_technique_fingers,
            "steps_done": 1,
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

    with _SESSIONS_LOCK:
        sessions[session_id] = {
            "session_dir": str(session_dir),
            "filename": "YouTube Video",
            "url": url,
            "status": SessionStatus.PENDING,
            "progress": "YouTube音声をダウンロード中...",
            "error": None,
            "tuning": request.tuning if request.tuning in TUNINGS else "standard",
            "guitar_type": request.guitar_type if request.guitar_type in ("auto", "steel", "nylon") else "auto",
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
    import sys
    _p = str(Path(__file__).parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from pipeline import run_pipeline

    session = sessions[session_id]
    session_dir = Path(session["session_dir"])
    wav_path = Path(session["wav_path"])
    tuning_name = session.get("tuning", "standard")

    STEP_MAP = {
        "beats": 1, "key": 1, "capo": 1,
        "demucs": 1, "preprocess": 1,
        "parallel": 2,
        "notes": 2, "spectral": 2,
        "filter": 3, "assign": 3, "note_filter": 3, "quantize": 3,
        "technique": 3, "technique_pm": 3, "technique_cnn": 3,
        "tuning_detect": 3, "chords": 3, "theory": 3,
        "musicxml": 4, "pdf": 4,
    }

    # ユーザー画面で見せるシンプルな進行メッセージ（デバッグログの混入を防ぐ）
    DISPLAY_TEXTS = {
        "beats": "ビート検出中...",
        "notes": "ノート検出中 (MoE+BP)...",
        "assign": "弦・フレット最適化中...",
        "musicxml": "TAB譜生成中...",
    }

    def progress_cb(step: str, msg: str):
        session["_current_step"] = step
        if step in DISPLAY_TEXTS:
            session["progress"] = DISPLAY_TEXTS[step]
            
        # steps_done only increases (never regresses) — critical for parallel execution
        mapped = STEP_MAP.get(step, session.get("steps_done", 0))
        current = session.get("steps_done", 0)
        if mapped > current:
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
            guitar_type=session.get("guitar_type", "auto"),
            transcription_profile=session.get("transcription_profile", "standard"),
            enable_technique_gp5=session.get("enable_technique_gp5", True),
            enable_technique_overlay=session.get("enable_technique_overlay", False),
            enable_technique_fingers=session.get("enable_technique_fingers", False),
            midi_path=Path(session["midi_path"]) if session.get("midi_path") else None,
        )

        session["status"] = SessionStatus.COMPLETED
        session["bpm"] = result["bpm"]
        session["time_signature"] = result.get("time_signature", "4/4")
        session["total_notes"] = result["total_notes"]
        session["key"] = result.get("key")
        session["capo"] = result.get("capo", 0)
        session["suggested_tuning"] = result.get("suggested_tuning")
        session["noise_gate"] = result.get("noise_gate", 0.10)  # BPM適応CUT初期値
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
            else:
                yield ": keep-alive\n\n"

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
        steps_done=s.get("steps_done", 0),
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
        time_signature=s.get("time_signature", "4/4"),
        filename=s.get("filename"),
        total_notes=s.get("total_notes"),
        tuning=s.get("tuning"),
        key=s.get("key"),
        capo=s.get("capo"),
        suggested_tuning=s.get("suggested_tuning"),
        noise_gate=s.get("noise_gate"),
        anchors=s.get("anchors", {})
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
    from starlette.responses import Response
    gp5_bytes = gp5_path.read_bytes()
    return Response(
        content=gp5_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": cd,
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/result/{session_id}/notes")
async def get_notes(session_id: str):
    """ノートデータを返す（カーソル同期用 — 各ノートのstart時刻を含む）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])
    assigned_path = session_dir / "notes_assigned.json"
    if not assigned_path.exists():
        raise HTTPException(status_code=404, detail="Notes not available")
    with open(assigned_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = data if isinstance(data, list) else data.get("notes", [])
    
    can_undo = "anchor_history" in s and s.get("anchor_history_index", -1) > 0
    can_redo = "anchor_history" in s and s.get("anchor_history_index", -1) < len(s["anchor_history"]) - 1
    
    return {
        "notes": notes,
        "can_undo": can_undo,
        "can_redo": can_redo
    }

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


def separate_melody_backing(notes: list, beats: list, beats_per_bar: int = 4) -> tuple[list, list]:
    if not notes or not beats:
        return notes.copy(), []
    
    import numpy as np
    beats_arr = np.array(beats)
    
    from collections import defaultdict
    beat_groups = defaultdict(list)
    
    for n in notes:
        t = float(n.get("start", n.get("start_time", 0.0)))
        idx = int(np.searchsorted(beats_arr, t, side='right')) - 1
        idx = max(0, min(idx, len(beats_arr) - 1))
        beat_groups[idx].append(n)
        
    melody_notes = []
    backing_notes = []
    
    for beat_idx, group in beat_groups.items():
        candidates = []
        for n in group:
            s = int(n.get("string", 1))
            p = int(n.get("pitch", 60))
            if s <= 3 and p > 52:
                candidates.append(n)
        
        if candidates:
            melody_note = max(candidates, key=lambda x: int(x.get("pitch", 0)))
            for n in group:
                if n is melody_note:
                    melody_notes.append(n)
                else:
                    backing_notes.append(n)
        else:
            for n in group:
                backing_notes.append(n)
                
    melody_notes.sort(key=lambda x: float(x.get("start", 0)))
    backing_notes.sort(key=lambda x: float(x.get("start", 0)))
    return melody_notes, backing_notes


def _patch_gp5_note(session_id: str, note_data: dict, old_fret: int, old_string: int, new_fret: int, new_string: int, new_pitch: int):
    """既存GP5ファイルの該当ノートだけを直接書き換える（全体再生成を回避）"""
    try:
        import guitarpro as gp
        s = sessions[session_id]
        session_dir = Path(s["session_dir"])
        gp5_path = session_dir / "tab.gp5"
        if not gp5_path.exists():
            print(f"[_patch_gp5_note] GP5 file not found, falling back to full regeneration")
            return False

        song = gp.parse(str(gp5_path))

        target_start = float(note_data.get("start", note_data.get("start_time", 0)))
        target_bar = note_data.get("bar")
        target_beat_pos = note_data.get("beat_pos_in_bar", note_data.get("beat_pos"))

        patched = False
        for track in song.tracks:
            for measure in track.measures:
                for voice in measure.voices:
                    for beat in voice.beats:
                        for note in beat.notes:
                            # Match by old fret + old string
                            if note.value == old_fret and note.string == old_string:
                                # If we have bar info, check bar number
                                if target_bar is not None:
                                    measure_num = measure.number if hasattr(measure, 'number') else None
                                    header_num = measure.header.number if hasattr(measure, 'header') and hasattr(measure.header, 'number') else None
                                    current_bar = measure_num or header_num
                                    if current_bar is not None and current_bar != target_bar:
                                        continue
                                # Patch in place
                                note.value = new_fret
                                note.string = new_string
                                patched = True
                                print(f"[_patch_gp5_note] Patched note: fret {old_fret}→{new_fret}, string {old_string}→{new_string} in bar {target_bar}")
                                break
                        if patched:
                            break
                    if patched:
                        break
                if patched:
                    break
            if patched:
                break

        if patched:
            gp.write(song, str(gp5_path))
            # Also update GP4
            try:
                gp.write(song, str(session_dir / "tab.gp4"))
            except Exception:
                pass
            print(f"[_patch_gp5_note] GP5 patched successfully")
            return True
        else:
            print(f"[_patch_gp5_note] Could not find matching note (fret={old_fret}, string={old_string}, bar={target_bar}), falling back to full regeneration")
            return False
    except Exception as e:
        print(f"[_patch_gp5_note] Error: {e}")
        import traceback; traceback.print_exc()
        return False


def _regenerate_musicxml(session_id: str, notes: list,
                         tuning: list = None, noise_gate: float = None):
    """notes → tab.gp5 + tab.musicxml 再生成の共通関数"""
    # 量子化済みデータは start_time を持ち start が欠落している場合がある
    # gp_renderer / music_quantizer は n["start"] を参照するため、ここで補完する
    for n in notes:
        if "start" not in n and "start_time" in n:
            n["start"] = n["start_time"]

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
        with open(beats_path, "r", encoding="utf-8") as f:
            bd = json.load(f)
        beats = bd.get("beats", [])
        bpm = bd.get("bpm", bpm)
        time_sig = bd.get("time_signature", time_sig)
        rhythm_info = bd.get("rhythm_info")  # triplet/straight情報を復元

    # 拍子から beats_per_bar を取得
    beats_per_bar = 4
    if time_sig == "3/4":
        beats_per_bar = 3
    elif time_sig == "6/8":
        beats_per_bar = 6

    # メロディとバッキングに分離
    melody_notes, backing_notes = separate_melody_backing(notes, beats, beats_per_bar)

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
    final_note_entries = None
    try:
        from gp_renderer import notes_to_gp5
        gp5_bytes, final_note_entries = notes_to_gp5(
            melody_notes, backing_notes=backing_notes, beats=beats, bpm=bpm, title=title,
            tuning=tuning, time_signature=time_sig,
            rhythm_info=rhythm_info, noise_gate=gate,
            return_entries=True,
        )
        with open(session_dir / "tab.gp5", "wb") as f:
            f.write(gp5_bytes)
        
        # 実際にGP5に書き込まれた、量子化・位置情報（bar, beat_pos）付きの最終ノート情報を notes_assigned.json に保存
        if final_note_entries is not None:
            with open(session_dir / "notes_assigned.json", "w", encoding="utf-8") as f:
                json.dump(final_note_entries, f, ensure_ascii=False, indent=2)
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
        backing_notes=backing_notes,
        title=title,
        tuning=tuning,
        time_signature=time_sig,
    )
    if noise_gate is not None:
        kwargs["noise_gate"] = noise_gate
    xml_content, tech_map = notes_to_tab_musicxml(melody_notes, **kwargs)

    with open(session_dir / "tab.musicxml", "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Delete stale PDF so it gets regenerated on next request
    pdf_path = session_dir / "tab.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    return xml_content, tech_map


class CutRequest(BaseModel):
    noise_gate: float = 0.0


@app.post("/result/{session_id}/cut")
async def cut_noise(session_id: str, request: CutRequest):
    """ノイズゲート(CUT)のみ変更 — 弦割り当て再実行なし、GP5のみ再生成（高速）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    session_dir = Path(s["session_dir"])

    original_path = session_dir / "notes_assigned_original.json"
    assigned_path = session_dir / "notes_assigned.json"
    notes_path = original_path if original_path.exists() else assigned_path
    if not notes_path.exists():
        notes_path = session_dir / "notes.json"

    if not notes_path.exists():
        raise HTTPException(status_code=404, detail="Notes data not found. Run analysis first.")

    with open(notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)

    notes = notes_data if isinstance(notes_data, list) else notes_data.get("notes", [])

    for n in notes:
        if "start" not in n and "start_time" in n:
            n["start"] = n["start_time"]

    _regenerate_musicxml(session_id, notes, noise_gate=request.noise_gate)
    s["noise_gate"] = request.noise_gate

    from gp_renderer import _filter_noise
    filtered_notes = _filter_noise(notes, request.noise_gate)

    with open(assigned_path, "w", encoding="utf-8") as f:
        json.dump(filtered_notes, f, ensure_ascii=False, indent=2)
    
    if original_path.exists():
        with open(original_path, "w", encoding="utf-8") as f:
            json.dump(filtered_notes, f, ensure_ascii=False, indent=2)

    filtered_count = len(filtered_notes)
    s["total_notes"] = filtered_count
    save_session(session_id)
    return {"status": "ok", "noise_gate": request.noise_gate, "total_notes": filtered_count}


class RetuneRequest(BaseModel):
    tuning: str
    capo: Optional[int] = 0
    noise_gate: Optional[float] = 0.0  # デフォルト0（CUTなし）


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

    # 量子化済みデータは start_time を持ち start が欠落している場合がある
    # music_quantizer / gp_renderer は n["start"] を参照するため、ここで補完する
    for n in notes:
        if "start" not in n and "start_time" in n:
            n["start"] = n["start_time"]

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


    # Re-run string assignment (with chords and guitar_type to preserve fingering engine logic)
    from string_assigner import assign_strings_dp
    chords = []
    chords_path = session_dir / "chords.json"
    if chords_path.exists():
        try:
            with open(chords_path, "r", encoding="utf-8") as f:
                chords = json.load(f)
        except Exception:
            pass
    guitar_type = s.get("guitar_type", "auto")
    key = s.get("key")

    notes = assign_strings_dp(
        notes,
        tuning=capo_tuning,
        chords=chords,
        guitar_type=guitar_type,
        key=key,
    )

    # フレットクランプ: パイプラインと同等の上限制約 (MAX_FRETを12から14に緩和)
    MAX_FRET = 14
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
        notes = assign_fingers(notes, detected_key=s.get("key"))
    except Exception as e:
        print(f"[retune] 指番号割り当てスキップ: {e}")

    # Save reassigned notes (only assigned_path, keep original_path as-is)
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
MAX_HISTORY = 50

def _init_anchor_history_if_needed(s: dict):
    if "anchor_history" not in s:
        s["anchor_history"] = [copy.deepcopy(s.get("anchors", {}))]
        s["anchor_history_index"] = 0

def _push_anchor_history(s: dict):
    """現在の anchors の状態を履歴に保存する"""
    _init_anchor_history_if_needed(s)
        
    history = s["anchor_history"]
    index = s["anchor_history_index"]
    
    # もし過去に戻っている状態で新たな操作が起きたら、未来の履歴を捨てる
    if index < len(history) - 1:
        history = history[:index + 1]
        s["anchor_history"] = history
        
    # 現在の状態をpush
    history.append(copy.deepcopy(s.get("anchors", {})))
    
    # 上限管理
    if len(history) > MAX_HISTORY:
        history.pop(0)
        index -= 1
        s["anchor_history"] = history
        
    s["anchor_history_index"] = len(history) - 1

def _inject_anchors_flag(notes, anchors):
    for n in notes:
        k = f"{int(n.get('pitch', 0))}_{round(float(n.get('start', n.get('start_time', 0.0))), 3)}"
        if k in anchors:
            n["_is_anchor"] = True
        else:
            n.pop("_is_anchor", None)
    return notes

class NoteEditRequest(BaseModel):
    fret: Optional[int] = None
    string: Optional[int] = None
    finger: Optional[int] = None
    delete: Optional[bool] = False
    anchor: Optional[bool] = None        # アンカーとして固定するかどうか（falseなら解除）
    start_time: Optional[float] = None   # 時刻ベース検索用
    old_fret: Optional[int] = None       # 元のフレット（照合用）


@app.patch("/result/{session_id}/notes/{note_index}")
async def edit_note(session_id: str, note_index: int, request: NoteEditRequest):
    """ノートを編集（フレット/弦変更 or 削除）→ MusicXML再生成"""
    print(f"[edit_note] session={session_id}, note_index={note_index}, fret={request.fret}, string={request.string}, delete={request.delete}, start_time={request.start_time}, old_fret={request.old_fret}")
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = sessions[session_id]
    if "anchors" not in s:
        s["anchors"] = {}
    _init_anchor_history_if_needed(s)
    session_dir = Path(s["session_dir"])
    assigned_path = session_dir / "notes_assigned.json"
    if not assigned_path.exists():
        raise HTTPException(status_code=404, detail="Notes not found")
    
    with open(assigned_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    # 時刻ベース検索: start_time + string で正確なノートを特定
    actual_index = note_index
    if request.start_time is not None and request.string is not None:
        best_idx = -1
        best_dist = float('inf')
        target_str = request.string if request.string else (request.old_fret if request.old_fret else None)
        for i, n in enumerate(notes):
            if int(n.get('string', 0)) == int(request.string):
                d = abs(n.get('start', 0) - request.start_time)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        if best_idx >= 0 and best_dist < 2.0:  # 2秒以内
            actual_index = best_idx
            print(f"[edit_note] Time-based match: note[{actual_index}] start={notes[actual_index].get('start')}, dist={best_dist:.3f}s")
        else:
            print(f"[edit_note] WARNING: No time-based match found, using index {note_index}")

    if actual_index < 0 or actual_index >= len(notes):
        raise HTTPException(status_code=400, detail=f"Invalid note index: {actual_index}")

    note_to_sync = dict(notes[actual_index])
    if request.delete:
        notes.pop(actual_index)
        action = "deleted"
    else:
        note = notes[actual_index]
        old_val = f"fret={note.get('fret')} string={note.get('string')} pitch={note.get('pitch')}"
        if request.fret is not None:
            note["fret"] = request.fret
            note["fixed_fret"] = request.fret
        if request.string is not None:
            note["string"] = request.string
            note["fixed_string"] = request.string
        if request.finger is not None:
            note["left_hand_finger"] = request.finger
        # pitch再計算: tuning[6-string] + fret (+ capo)
        tuning_name = s.get("tuning", "standard")
        tuning_arr = TUNINGS.get(tuning_name, TUNINGS["standard"])
        capo_val = s.get("capo", 0) or 0
        new_string = int(note.get("string", 1))
        new_fret = int(note.get("fret", 0))
        if 1 <= new_string <= 6:
            open_pitch = tuning_arr[6 - new_string] + capo_val
            note["pitch"] = open_pitch + new_fret
            
        note_key_str = f"{int(note.get('pitch', 0))}_{round(float(note.get('start', note.get('start_time', 0.0))), 3)}"
        
        if request.anchor is False:
            if note_key_str in s["anchors"]:
                del s["anchors"][note_key_str]
        else:
            anchor_data = {
                "string": note["string"],
                "fret": note["fret"]
            }
            if request.finger is not None:
                anchor_data["finger"] = request.finger
            elif "left_hand_finger" in note:
                anchor_data["finger"] = note["left_hand_finger"]
            s["anchors"][note_key_str] = anchor_data
            
        _push_anchor_history(s)

        action = f"edited [{old_val}] → fret={note.get('fret')} string={note.get('string')} pitch={note.get('pitch')}"

    notes = _inject_anchors_flag(notes, s["anchors"])

    with open(assigned_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    # Also apply the edit/deletion to notes_assigned_original.json to keep them in sync
    original_path = session_dir / "notes_assigned_original.json"
    if original_path.exists():
        try:
            with open(original_path, "r", encoding="utf-8") as f:
                orig_notes = json.load(f)
            
            target_start = note_to_sync.get("start")
            target_pitch = int(note_to_sync.get("pitch", 0))
            
            match_idx = -1
            best_dist = float('inf')
            for i, n in enumerate(orig_notes):
                n_start = n.get("start", n.get("start_time", 0.0))
                d = abs(n_start - target_start)
                if d < 0.15 and int(n.get("pitch", 0)) == target_pitch:
                    if d < best_dist:
                        best_dist = d
                        match_idx = i
            
            if match_idx >= 0:
                if request.delete:
                    orig_notes.pop(match_idx)
                else:
                    orig_note = orig_notes[match_idx]
                    if request.fret is not None:
                        orig_note["fret"] = request.fret
                        orig_note["fixed_fret"] = request.fret
                    if request.string is not None:
                        orig_note["string"] = request.string
                        orig_note["fixed_string"] = request.string
                    # Recalculate pitch for original note
                    tuning_name = s.get("tuning", "standard")
                    tuning_arr = TUNINGS.get(tuning_name, TUNINGS["standard"])
                    capo_val = s.get("capo", 0) or 0
                    new_string = int(orig_note.get("string", 1))
                    new_fret = int(orig_note.get("fret", 0))
                    if 1 <= new_string <= 6:
                        open_pitch = tuning_arr[6 - new_string] + capo_val
                        orig_note["pitch"] = open_pitch + new_fret
                
                with open(original_path, "w", encoding="utf-8") as f:
                    json.dump(orig_notes, f, ensure_ascii=False, indent=2)
                print(f"[edit_note] Synced with notes_assigned_original.json at index {match_idx}")
            else:
                print(f"[edit_note] WARNING: Could not find matching note in notes_assigned_original.json for sync")
        except Exception as e:
            print(f"[edit_note] ERROR syncing with notes_assigned_original.json: {e}")

    # Verify write
    with open(assigned_path, "r", encoding="utf-8") as f:
        verify = json.load(f)
    if not request.delete and actual_index < len(verify):
        v = verify[actual_index]
        print(f"[edit_note] VERIFY: note[{actual_index}] fret={v.get('fret')}, string={v.get('string')}, pitch={v.get('pitch')}")

    try:
        if request.delete:
            # 削除の場合は全体再生成が必要
            _regenerate_musicxml(session_id, notes)
        else:
            # Viterbi再計算 (Human-in-the-Loop)
            try:
                from string_assigner import assign_strings_dp
                from finger_assigner import assign_fingers
                tuning_name = s.get("tuning", "standard")
                tuning_arr = TUNINGS.get(tuning_name, TUNINGS["standard"])
                guitar_type = s.get("guitar_type", "auto")
                
                forced_positions = {}
                forced_fingers = {}
                for k, v in s.get("anchors", {}).items():
                    parts = k.split("_")
                    if len(parts) == 2:
                        try:
                            pitch = int(parts[0])
                            start = float(parts[1])
                            if "string" in v and "fret" in v:
                                forced_positions[(pitch, start)] = (v["string"], v["fret"])
                            if "finger" in v:
                                forced_fingers[(pitch, start)] = v["finger"]
                        except ValueError:
                            pass
                
                print(f"[edit_note] Running assign_strings_dp for Human-in-the-Loop Viterbi... (Anchors: {len(forced_positions)})")
                notes = assign_strings_dp(notes, tuning=tuning_arr, max_fret=24, guitar_type=guitar_type, key=s.get("key"), forced_positions=forced_positions)
                notes = assign_fingers(notes, detected_key=s.get("key"), forced_fingers=forced_fingers)
                notes = _inject_anchors_flag(notes, s["anchors"])
                
                with open(assigned_path, "w", encoding="utf-8") as f:
                    json.dump(notes, f, ensure_ascii=False, indent=2)
                
                original_path = session_dir / "notes_assigned_original.json"
                if original_path.exists():
                    with open(original_path, "w", encoding="utf-8") as f:
                        json.dump(notes, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[edit_note] Viterbi再計算エラー: {e}")
                
            # 全体再生成 (複数ノートが変わる可能性があるため)
            _regenerate_musicxml(session_id, notes)
    except Exception as e:
        print(f"[edit_note] Regeneration failed: {e}")
        import traceback; traceback.print_exc()

    # Verify file not overwritten by _regenerate_musicxml
    with open(assigned_path, "r", encoding="utf-8") as f:
        verify2 = json.load(f)
    if not request.delete and actual_index < len(verify2):
        v2 = verify2[actual_index]
        print(f"[edit_note] AFTER REGEN: note[{actual_index}] fret={v2.get('fret')}, string={v2.get('string')}, pitch={v2.get('pitch')}")

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

    original_path = session_dir / "notes_assigned_original.json"
    if original_path.exists():
        try:
            with open(original_path, "r", encoding="utf-8") as f:
                orig_notes = json.load(f)
            
            insert_idx_orig = 0
            for i, n in enumerate(orig_notes):
                if float(n.get("start", 0)) > request.start:
                    insert_idx_orig = i
                    break
                insert_idx_orig = i + 1
            orig_notes.insert(insert_idx_orig, new_note)
            
            with open(original_path, "w", encoding="utf-8") as f:
                json.dump(orig_notes, f, ensure_ascii=False, indent=2)
            print(f"[add_note] Synced with notes_assigned_original.json at index {insert_idx_orig}")
        except Exception as e:
            print(f"[add_note] ERROR syncing with notes_assigned_original.json: {e}")

    _regenerate_musicxml(session_id, notes)

    s["total_notes"] = len(notes)
    save_session(session_id)

    return {"status": "ok", "action": "added", "note_index": insert_idx, "total_notes": len(notes)}



@app.post("/result/{session_id}/anchors/reset")
async def reset_anchors(session_id: str):
    """すべてのアンカーをクリアし、AIのデフォルト推論（Viterbi）に戻す"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = sessions[session_id]
    _init_anchor_history_if_needed(s)
    s["anchors"] = {}
    _push_anchor_history(s)
    save_session(session_id)
    
    session_dir = Path(s["session_dir"])
    assigned_path = session_dir / "notes_assigned.json"
    if not assigned_path.exists():
        raise HTTPException(status_code=404, detail="Notes not found")
        
    with open(assigned_path, "r", encoding="utf-8") as f:
        notes = json.load(f)
        
    try:
        from string_assigner import assign_strings_dp
        from finger_assigner import assign_fingers
        tuning_name = s.get("tuning", "standard")
        tuning_arr = TUNINGS.get(tuning_name, TUNINGS["standard"])
        guitar_type = s.get("guitar_type", "auto")
        
        print(f"[reset_anchors] Running default assign_strings_dp...")
        notes = assign_strings_dp(notes, tuning=tuning_arr, max_fret=24, guitar_type=guitar_type, key=s.get("key"))
        notes = assign_fingers(notes, detected_key=s.get("key"))
        notes = _inject_anchors_flag(notes, s["anchors"])
        
        with open(assigned_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
            
        original_path = session_dir / "notes_assigned_original.json"
        if original_path.exists():
            with open(original_path, "w", encoding="utf-8") as f:
                json.dump(notes, f, ensure_ascii=False, indent=2)
                
        _regenerate_musicxml(session_id, notes)
    except Exception as e:
        print(f"[reset_anchors] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="Re-optimization failed")

    return {"status": "ok", "action": "reset_anchors"}


@app.post("/result/{session_id}/undo")
async def undo_anchors(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = sessions[session_id]
    if "anchor_history" not in s or s.get("anchor_history_index", -1) <= 0:
        return {"status": "ok", "action": "undo", "message": "Nothing to undo"}
        
    s["anchor_history_index"] -= 1
    s["anchors"] = copy.deepcopy(s["anchor_history"][s["anchor_history_index"]])
    save_session(session_id)
    
    _recompute_anchors_logic(s)
    return {"status": "ok", "action": "undo"}


@app.post("/result/{session_id}/redo")
async def redo_anchors(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = sessions[session_id]
    if "anchor_history" not in s or s.get("anchor_history_index", -1) >= len(s["anchor_history"]) - 1:
        return {"status": "ok", "action": "redo", "message": "Nothing to redo"}
        
    s["anchor_history_index"] += 1
    s["anchors"] = copy.deepcopy(s["anchor_history"][s["anchor_history_index"]])
    save_session(session_id)
    
    _recompute_anchors_logic(s)
    return {"status": "ok", "action": "redo"}


def _recompute_anchors_logic(s: dict):
    session_dir = Path(s["session_dir"])
    assigned_path = session_dir / "notes_assigned.json"
    original_path = session_dir / "notes_assigned_original.json"
    if not original_path.exists():
        return
        
    with open(original_path, "r", encoding="utf-8") as f:
        notes = json.load(f)
        
    tuning_name = s.get("suggested_tuning", "standard")
    tuning_arr = TUNINGS.get(tuning_name, TUNINGS["standard"])
    guitar_type = s.get("guitar_type", "auto")
    
    forced_positions = {}
    forced_fingers = {}
    for k, v in s.get("anchors", {}).items():
        parts = k.split("_")
        if len(parts) == 2:
            try:
                pitch = int(parts[0])
                start = float(parts[1])
                if "string" in v and "fret" in v:
                    forced_positions[(pitch, start)] = (v["string"], v["fret"])
                if "finger" in v:
                    forced_fingers[(pitch, start)] = v["finger"]
            except:
                pass
                
    notes = assign_strings_dp(notes, tuning=tuning_arr, max_fret=24, guitar_type=guitar_type, key=s.get("key"), forced_positions=forced_positions)
    notes = assign_fingers(notes, detected_key=s.get("key"), forced_fingers=forced_fingers)
    notes = _inject_anchors_flag(notes, s["anchors"])
    
    with open(assigned_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)
        
    session_id = session_dir.name
    try:
        _regenerate_musicxml(session_id, notes)
    except:
        pass


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
    try:
        if not file_path.resolve().is_relative_to(session_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/sessions")
async def get_sessions():
    history = []
    with _SESSIONS_LOCK:
        sessions_copy = list(sessions.items())
    for sid, s in sorted(sessions_copy, key=lambda x: x[0], reverse=True):
        history.append({
            "session_id": sid,
            "filename": s.get("filename", "Unknown"),
            "status": s.get("status"),
            "bpm": s.get("bpm"),
            "total_notes": s.get("total_notes"),
        })
    return history


@app.post("/api/transcribe_midi")
async def api_transcribe_midi(
    file: UploadFile = File(...),
    tuning: str = Form("standard"),
    style_profile: str = Form("classic"),
    background_tasks: BackgroundTasks = None
):
    """
    [TASK-904] MIDIファイルを直接入力とし、SYMBOLIC_MIDI_BYPASS (Transformer V3)
    を用いてTAB譜 (GP5) を生成する。
    """
    if not file.filename.lower().endswith(('.mid', '.midi')):
        raise HTTPException(status_code=400, detail="Only .mid and .midi files are accepted for MIDI bypass.")

    session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    midi_path = session_dir / file.filename
    with open(midi_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ダミーWAV作成
    dummy_wav = session_dir / "converted.wav"
    sr = 22050
    t_arr = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    dummy_sig = 0.1 * np.sin(2 * np.pi * 220.0 * t_arr)
    import soundfile as sf
    sf.write(str(dummy_wav), dummy_sig, sr)

    sessions[session_id] = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "filename": file.filename,
        "mode": "SYMBOLIC_MIDI_BYPASS",
        "status": "processing",
        "progress": 0.1,
        "step": "midi_bypass_init",
        "tuning": tuning,
        "created_at": time.time()
    }

    def _process():
        try:
            from pipeline import run_pipeline
            run_pipeline(
                session_id, session_dir, dummy_wav,
                tuning_name=tuning,
                transcription_profile=style_profile,
                midi_path=midi_path
            )
            sessions[session_id]["status"] = "complete"
            sessions[session_id]["progress"] = 1.0
            sessions[session_id]["step"] = "done"
            sessions[session_id]["gp5_url"] = f"/files/{session_id}/tab.gp5"
        except Exception as e:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["error"] = str(e)

    if background_tasks:
        background_tasks.add_task(_process)
    else:
        _process()

    return {
        "session_id": session_id,
        "mode": "SYMBOLIC_MIDI_BYPASS",
        "status": "processing",
        "filename": file.filename
    }


@app.post("/api/refinger")
async def api_refinger(
    file: UploadFile = File(...),
    tuning: str = Form("standard")
):
    """
    [TASK-904] GP5 / MusicXML ファイルを直接入力とし、Voiceとアーティキュレーションを
    100%保持したまま Transformer V3 による運指最適化を実行する。
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ('.gp5', '.gp', '.gpx', '.xml', '.musicxml'):
        raise HTTPException(status_code=400, detail="Only .gp5, .gp, and .musicxml files are accepted for refingering.")

    session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-refinger-" + uuid.uuid4().hex[:6]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    input_path = session_dir / file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_gp5_name = f"{Path(file.filename).stem}_refingered.gp5"
    output_path = session_dir / output_gp5_name

    try:
        from refingering_engine import refinger_gp5
        refinger_res = refinger_gp5(str(input_path), str(output_path))
        
        orig_mov = refinger_res["original_ergonomic_cost"]["total_movement_frets"]
        opt_mov = refinger_res["optimized_ergonomic_cost"]["total_movement_frets"]
        red_ratio = round((orig_mov - opt_mov) / max(1.0, orig_mov) * 100.0, 1) if opt_mov < orig_mov else 0.0

        sessions[session_id] = {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "filename": file.filename,
            "mode": "NATIVE_GP5_REFINGERING",
            "status": "complete",
            "progress": 1.0,
            "refinger_metrics": refinger_res,
            "reduction_percent": red_ratio,
            "gp5_url": f"/files/{session_id}/{output_gp5_name}"
        }

        return {
            "session_id": session_id,
            "status": "complete",
            "filename": file.filename,
            "output_filename": output_gp5_name,
            "download_url": f"/files/{session_id}/{output_gp5_name}",
            "ergonomic_metrics": {
                "original_movement_frets": orig_mov,
                "optimized_movement_frets": opt_mov,
                "movement_reduction_percent": f"-{red_ratio}%" if red_ratio > 0 else f"{red_ratio}%",
                "exact_matches_with_original": refinger_res["exact_matches_with_original_gp5"],
                "match_rate": f"{refinger_res['string_fret_match_rate']:.1%}",
                "refingered_notes_count": refinger_res["refingered_notes_count"],
                "preserved_voices_count": refinger_res["preserved_voices_count"]
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Refingering failed: {str(e)}")


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

    uvicorn.run(app, host="0.0.0.0", port=8002)
