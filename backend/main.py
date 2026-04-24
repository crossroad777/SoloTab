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

# NumPy 2.0+ patch for madmom
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'bool'): np.bool = bool

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# venv312 の Python
VENV_DIR = PROJECT_ROOT.parent / "nextchord" / "venv312"
PYTHON_PATH = str(VENV_DIR / "Scripts" / "python.exe")

# FFMPEGパス
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT.parent / "nextchord" / ".env")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFMPEG_BIN_DIR = str(Path(FFMPEG_PATH).parent)
if FFMPEG_BIN_DIR and FFMPEG_BIN_DIR not in os.environ.get("PATH", ""):
    # 最前列に追加して優先度を上げる
    os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    print(f"[SoloTab] Added ffmpeg to PATH: {FFMPEG_BIN_DIR}")

# yt-dlp パス
YT_DLP_PATH = os.getenv("YT_DLP_PATH", "yt-dlp")
if not shutil.which(YT_DLP_PATH):
    venv_yt = VENV_DIR / "Scripts" / "yt-dlp.exe"
    if venv_yt.exists():
        YT_DLP_PATH = str(venv_yt)
print(f"[SoloTab] FFMPEG: {FFMPEG_PATH}, yt-dlp: {YT_DLP_PATH}")

# Available tunings (for validation)
from string_assigner import TUNINGS

# Uploads
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Models (lazy load) ---
# Removed global model definitions here to speed up startup time.
# Models will be imported and loaded on-demand in the pipeline tasks.

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


# --- Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...),
                       tuning: str = Form("standard"),
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
    from pipeline import run_pipeline

    session = sessions[session_id]
    session_dir = Path(session["session_dir"])
    wav_path = Path(session["wav_path"])
    tuning_name = session.get("tuning", "standard")

    # パイプライン内部ステップ → フロントエンド4ステップのマッピング
    STEP_MAP = {
        "beats": 0, "key": 0, "capo": 0,
        "demucs": 1, "preprocess": 1,
        "notes": 1, "spectral": 1,
        "filter": 2, "assign": 2, "note_filter": 2, "quantize": 2,
        "technique": 2, "technique_pm": 2, "tuning_detect": 2, "chords": 2,
        "musicxml": 3,
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
    # ASCII文字のみの場合はそのまま、非ASCII文字がある場合はURLエンコード
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

    import json as json_mod
    import shutil

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
        notes_data = json_mod.load(f)

    # notes_assigned.jsonはリスト直接、notes.jsonは{"notes": [...]}
    import copy
    notes = copy.deepcopy(notes_data if isinstance(notes_data, list) else notes_data.get("notes", notes_data))
    tuning = TUNINGS[tuning_name]

    # カポ対応: tuningにカポ分を加算
    # request.capo=0でも有効（カポ解除）にするためis not Noneチェック
    capo = request.capo if request.capo is not None else s.get("capo", 0)
    if capo is None:
        capo = 0
    if capo > 0:
        capo_tuning = [p + capo for p in tuning]
    else:
        capo_tuning = tuning

    # Re-run string assignment
    from string_assigner import assign_strings_dp
    from tab_renderer import notes_to_tab_musicxml

    notes = assign_strings_dp(notes, tuning=capo_tuning)

    # Save reassigned notes（オリジナルは保持、表示用のみ上書き）
    with open(assigned_path, "w", encoding="utf-8") as f:
        json_mod.dump(notes, f, ensure_ascii=False, indent=2)

    # Load beats
    beats_path = session_dir / "beats.json"
    beats = []
    bpm = s.get("bpm", 120)
    time_sig = s.get("time_signature", "4/4")
    if beats_path.exists():
        with open(beats_path, "r") as f:
            beat_data = json_mod.load(f)
        beats = beat_data.get("beats", [])
        bpm = beat_data.get("bpm", bpm)
        time_sig = beat_data.get("time_signature", time_sig)

    # Re-generate MusicXML (カポ適用済みtuningを使用)
    xml_content, tech_map = notes_to_tab_musicxml(
        notes, beats=beats, bpm=bpm,
        title=s.get("filename", session_id),
        tuning=capo_tuning,
        time_signature=time_sig,
        noise_gate=request.noise_gate
    )
    with open(session_dir / "tab.musicxml", "w", encoding="utf-8") as f:
        f.write(xml_content)
    # techniques.jsonはカポ/チューニングに依存しない → オリジナルを保持
    tech_original = session_dir / "techniques_original.json"
    tech_current = session_dir / "techniques.json"
    if not tech_original.exists() and tech_current.exists():
        shutil.copy2(tech_current, tech_original)
    # MusicXML再生成で生まれたtech_mapは使わず、オリジナルを復元
    if tech_original.exists():
        shutil.copy2(tech_original, tech_current)

    # Update session
    s["tuning"] = tuning_name
    s["capo"] = capo
    s["total_notes"] = len(notes)
    save_session(session_id)

    return {"status": "ok", "tuning": tuning_name, "capo": capo, "total_notes": len(notes)}


class NoteEditRequest(BaseModel):
    noteIndex: int
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
    
    import json as json_mod
    with open(assigned_path, "r", encoding="utf-8") as f:
        notes = json_mod.load(f)
    
    if note_index < 0 or note_index >= len(notes):
        raise HTTPException(status_code=400, detail=f"Invalid note index: {note_index}")
    
    if request.delete:
        deleted = notes.pop(note_index)
        action = "deleted"
    else:
        note = notes[note_index]
        if request.fret is not None:
            note["fret"] = request.fret
        if request.string is not None:
            note["string"] = request.string
        action = f"edited fret={note.get('fret')} string={note.get('string')}"
    
    # Save updated notes
    with open(assigned_path, "w", encoding="utf-8") as f:
        json_mod.dump(notes, f, ensure_ascii=False, indent=2)
    
    # Regenerate MusicXML
    from tab_renderer import notes_to_tab_musicxml
    tuning_name = s.get("tuning", "standard")
    tuning = TUNINGS.get(tuning_name, TUNINGS["standard"])
    capo = s.get("capo", 0)
    if capo > 0:
        tuning = [p + capo for p in tuning]
    
    beats_path = session_dir / "beats.json"
    beats, bpm = [], s.get("bpm", 120)
    time_sig = s.get("time_signature", "4/4")
    if beats_path.exists():
        with open(beats_path, "r") as f:
            bd = json_mod.load(f)
        beats = bd.get("beats", [])
        bpm = bd.get("bpm", bpm)
        time_sig = bd.get("time_signature", time_sig)
    
    xml_content, _ = notes_to_tab_musicxml(
        notes, beats=beats, bpm=bpm,
        title=s.get("filename", session_id),
        tuning=tuning,
        time_signature=time_sig,
    )
    with open(session_dir / "tab.musicxml", "w", encoding="utf-8") as f:
        f.write(xml_content)
    
    s["total_notes"] = len(notes)
    save_session(session_id)
    
    return {"status": "ok", "action": action, "total_notes": len(notes)}


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
