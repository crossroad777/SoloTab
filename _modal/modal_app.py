"""
SoloTab V2.0 — Modal デプロイメント
====================================
フルパイプライン: librosa ビート検出 + Pure MoE 推論 + MusicXML 生成

使い方:
  python -m modal deploy _modal/modal_app.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Image: 依存関係 + ソースコード + モデルの重み
# ---------------------------------------------------------------------------

_MODEL_BASE = r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output"
_EXPERTS = [
    "finetuned_martin_finger_guitarset_ft",
    "finetuned_taylor_finger_guitarset_ft",
    "finetuned_luthier_finger_guitarset_ft",
    "finetuned_martin_pick_guitarset_ft",
    "finetuned_taylor_pick_guitarset_ft",
    "finetuned_luthier_pick_guitarset_ft",
]

solotab_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "fonts-noto-cjk")
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "numpy<2.0",
        "librosa==0.10.2",
        "scipy",
        "soundfile",
        "fastapi[standard]",
        "uvicorn[standard]",
        "python-multipart",
        "pydantic",
        "midiutil",
        "reportlab",
    )
    .add_local_dir(
        r"D:\Music\nextchord-solotab\backend",
        remote_path="/app/backend",
        copy=True,
        ignore=lambda p: any(str(p).replace("\\","/").startswith(d) for d in [
            "__pycache__","benchmark","uploads","logs","train","synth",
            "fretnet_models","ground_truth","models",
        ]) or str(p).endswith(".pyc"),
    )
    .add_local_dir(
        r"D:\Music\nextchord-solotab\music-transcription\python\model",
        remote_path="/app/music-transcription/python/model",
        copy=True,
    )
    .add_local_file(
        r"D:\Music\nextchord-solotab\music-transcription\python\config.py",
        remote_path="/app/music-transcription/python/config.py",
        copy=True,
    )
)

for expert in _EXPERTS:
    solotab_image = solotab_image.add_local_file(
        os.path.join(_MODEL_BASE, expert, "best_model.pth"),
        remote_path=f"/app/music-transcription/python/_processed_guitarset_data/training_output/{expert}/best_model.pth",
        copy=True,
    )

app = modal.App("solotab", image=solotab_image)

# セッションデータを永続化するための Volume
session_vol = modal.Volume.from_name("solotab-sessions", create_if_missing=True)


# ---------------------------------------------------------------------------
# ASGI エントリーポイント
# ---------------------------------------------------------------------------
@app.function(
    timeout=600,
    memory=4096,
    cpu=2.0,
    min_containers=0,
    volumes={"/data": session_vol},
)
@modal.concurrent(max_inputs=3)
@modal.asgi_app()
def solotab_api():
    import sys
    import numpy as np
    for attr in ('int','float','complex','bool'):
        if not hasattr(np, attr):
            setattr(np, attr, eval(attr))

    sys.path.insert(0, "/app/backend")
    sys.path.insert(0, "/app/music-transcription/python")
    os.environ["FFMPEG_PATH"] = "ffmpeg"

    from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    from contextlib import asynccontextmanager
    import json, shutil, subprocess, datetime as dt, copy
    from typing import Optional
    from enum import Enum
    from pathlib import Path
    from solotab_utils import TUNINGS, _to_native

    # Volume 上にセッションデータを保存（コンテナ再起動後も残る）
    UPLOAD_DIR = Path("/data/sessions")
    UPLOAD_DIR.mkdir(exist_ok=True)

    class SS(str, Enum):
        PENDING="pending"; PROCESSING="processing"; COMPLETED="completed"; FAILED="failed"

    sessions = {}

    def save(sid):
        if sid in sessions:
            sd = Path(sessions[sid]["session_dir"])
            (sd/"session.json").write_text(json.dumps(sessions[sid], ensure_ascii=False, indent=2), encoding="utf-8")
            session_vol.commit()

    def load_session(sid):
        """Volume からセッションを復元"""
        if sid in sessions:
            return sessions[sid]
        sd = UPLOAD_DIR / sid
        sp = sd / "session.json"
        if sp.exists():
            s = json.loads(sp.read_text(encoding="utf-8"))
            sessions[sid] = s
            return s
        return None

    def load_all_sessions():
        """起動時に Volume 上の全セッションを読み込む"""
        if not UPLOAD_DIR.exists():
            return
        for d in UPLOAD_DIR.iterdir():
            if d.is_dir():
                sp = d / "session.json"
                if sp.exists():
                    try:
                        s = json.loads(sp.read_text(encoding="utf-8"))
                        sessions[d.name] = s
                    except Exception:
                        pass

    load_all_sessions()

    @asynccontextmanager
    async def lifespan(_): yield

    fa = FastAPI(title="SoloTab API", version="2.0.0", lifespan=lifespan)
    fa.add_middleware(
        CORSMiddleware,
        allow_origins=["https://solotab.vercel.app", "http://localhost:5173", "http://localhost:3000", "http://localhost:8001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class UR(BaseModel):
        session_id: str; message: str; status: SS; audio_url: Optional[str]=None
    class SR(BaseModel):
        session_id: str; status: SS; progress: Optional[str]=None; error: Optional[str]=None; filename: Optional[str]=None
    class RR(BaseModel):
        session_id: str; status: SS; bpm: Optional[float]=None; filename: Optional[str]=None; total_notes: Optional[int]=None; tuning: Optional[str]=None; key: Optional[str]=None; capo: Optional[int]=None; suggested_tuning: Optional[str]=None
    class RTR(BaseModel):
        tuning: str; capo: Optional[int]=0; noise_gate: Optional[float]=0.30

    def _run_bg(sid):
        """Pure MoE パイプライン（librosa ビート検出 + MoE推論）"""
        s = sessions[sid]
        sd = Path(s["session_dir"])
        wav = Path(s["wav_path"])
        try:
            s["status"] = SS.PROCESSING; save(sid)

            s["progress"] = "ビート検出中..."; s["steps_done"] = 0; save(sid)
            from beat_detector import detect_beats
            beat_result = detect_beats(str(wav))
            beats_list = beat_result["beats"]
            bpm = beat_result["bpm"]
            time_sig = beat_result.get("time_signature", "4/4")
            (sd/"beats.json").write_text(json.dumps(beat_result, ensure_ascii=False), encoding="utf-8")

            s["progress"] = "MoE推論中..."; s["steps_done"] = 1; save(sid)
            from pure_moe_transcriber import transcribe_pure_moe
            notes = transcribe_pure_moe(str(wav), vote_threshold=5, onset_threshold=0.8)

            s["progress"] = "テクニック検出中..."; s["steps_done"] = 2
            try:
                from technique_detector import detect_techniques, add_techniques_to_musicxml_notes
                notes = detect_techniques(notes, bpm=bpm)
                notes = add_techniques_to_musicxml_notes(notes)
            except Exception as e:
                print(f"Technique detection skipped: {e}")

            (sd/"notes_assigned.json").write_text(json.dumps(_to_native(notes), ensure_ascii=False, indent=2), encoding="utf-8")

            s["progress"] = "TAB譜生成中..."; s["steps_done"] = 3
            from tab_renderer import notes_to_tab_musicxml
            tuning = TUNINGS.get(s.get("tuning","standard"), TUNINGS["standard"])
            title = s.get("filename", sid)
            ext = os.path.splitext(title)[1]
            if ext.lower() in {".mp3",".wav",".m4a",".flac",".ogg"}: title = title[:-len(ext)]

            xml_content, tech_map = notes_to_tab_musicxml(
                notes, beats=beats_list, bpm=bpm, title=title,
                tuning=tuning, time_signature=time_sig
            )
            (sd/"tab.musicxml").write_text(xml_content, encoding="utf-8")
            (sd/"techniques.json").write_text(json.dumps(_to_native(tech_map)), encoding="utf-8")

            s.update({"status":SS.COMPLETED,"total_notes":len(notes),"bpm":bpm,
                      "time_signature":time_sig,"progress":"解析完了","steps_done":4})
            save(sid)
        except Exception as e:
            import traceback; traceback.print_exc()
            s.update({"status":SS.FAILED,"error":str(e),"progress":"エラー"})
            save(sid)

    @fa.post("/upload", response_model=UR)
    async def upload(file: UploadFile=File(...), tuning:str=Form("standard"), background_tasks:BackgroundTasks=None):
        sid = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        sd = UPLOAD_DIR/sid; sd.mkdir(parents=True, exist_ok=True)
        ap = sd/file.filename
        ap.write_bytes(await file.read())
        wp = sd/"converted.wav"
        if ap.suffix.lower()!=".wav":
            subprocess.run(["ffmpeg","-y","-i",str(ap),"-ar","22050","-ac","1",str(wp)], check=True, capture_output=True)
        else:
            shutil.copy2(str(ap),str(wp))
        sessions[sid] = {"session_dir":str(sd),"filename":file.filename,"wav_path":str(wp),"status":SS.PENDING,"progress":"アップロード完了","error":None,"tuning":tuning if tuning in TUNINGS else "standard"}
        save(sid)
        background_tasks.add_task(_run_bg, sid)
        return UR(session_id=sid, message="解析を開始しました", status=SS.PENDING, audio_url=f"/files/{sid}/converted.wav")

    @fa.get("/status/{sid}/stream")
    async def stream(sid:str):
        import asyncio
        from starlette.responses import StreamingResponse
        async def g():
            last=None
            while True:
                s = load_session(sid)
                if s is None: yield f"data: {json.dumps({'status':'not_found'})}\n\n"; return
                cur={"status":s.get("status","pending"),"progress":s.get("progress",""),"filename":s.get("filename"),"steps_done":s.get("steps_done",0)}
                k=f"{cur['status']}:{cur['progress']}:{cur['steps_done']}"
                if k!=last: yield f"data: {json.dumps(cur,ensure_ascii=False)}\n\n"; last=k
                if cur["status"] in ("completed","failed"):
                    if cur["status"]=="failed": cur["error"]=s.get("error",""); yield f"data: {json.dumps(cur,ensure_ascii=False)}\n\n"
                    return
                await asyncio.sleep(0.8)
        return StreamingResponse(g(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","Connection":"keep-alive"})

    @fa.get("/status/{sid}",response_model=SR)
    async def status(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        return SR(session_id=sid,status=s["status"],progress=s.get("progress"),error=s.get("error"),filename=s.get("filename"))

    @fa.get("/result/{sid}",response_model=RR)
    async def result(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        if s["status"]!=SS.COMPLETED: raise HTTPException(202,"Not complete")
        return RR(session_id=sid,status=s["status"],bpm=s.get("bpm"),filename=s.get("filename"),total_notes=s.get("total_notes"),tuning=s.get("tuning"),key=s.get("key"),capo=s.get("capo"),suggested_tuning=s.get("suggested_tuning"))

    @fa.get("/result/{sid}/musicxml")
    async def musicxml(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        xp=Path(s["session_dir"])/"tab.musicxml"
        if not xp.exists(): raise HTTPException(404,"MusicXML not generated")
        from urllib.parse import quote
        raw_name = s.get("filename","tab").rsplit(".",1)[0] + ".musicxml"
        safe_name = quote(raw_name)
        return Response(
            content=xp.read_text(encoding="utf-8"),
            media_type="application/xml",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_name}"}
        )

    @fa.get("/result/{sid}/pdf")
    async def pdf(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        sd=Path(s["session_dir"])
        xp=sd/"tab.musicxml"
        if not xp.exists(): raise HTTPException(404,"MusicXML not generated")
        pp=sd/"tab.pdf"
        from pdf_renderer import musicxml_to_pdf
        title = s.get("filename","tab").rsplit(".",1)[0]
        musicxml_to_pdf(str(xp), str(pp), title=title)
        from urllib.parse import quote
        safe_name = quote(title + ".pdf")
        return FileResponse(pp, filename="tab.pdf", media_type="application/pdf",
                            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_name}"})

    @fa.get("/result/{sid}/notes")
    async def notes(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        np_=Path(s["session_dir"])/"notes_assigned.json"
        return {"notes":json.loads(np_.read_text(encoding="utf-8"))} if np_.exists() else {"notes":[]}

    @fa.get("/result/{sid}/techniques")
    async def techniques(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        tp=Path(s["session_dir"])/"techniques.json"
        return json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else []

    @fa.get("/result/{sid}/beats")
    async def beats(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        bp=Path(s["session_dir"])/"beats.json"
        return json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {"beats":[],"bpm":120}

    @fa.api_route("/files/{sid}/{fn}", methods=["GET","HEAD"])
    async def get_file(sid:str,fn:str):
        s = load_session(sid)
        sd = Path(s["session_dir"]) if s else UPLOAD_DIR/sid
        fp=sd/fn
        if not fp.exists(): raise HTTPException(404,"File not found")
        return FileResponse(fp)

    @fa.get("/sessions")
    async def get_sessions():
        load_all_sessions()
        return [{"session_id":k,"filename":v.get("filename","?"),"status":v.get("status"),"bpm":v.get("bpm"),"total_notes":v.get("total_notes")} for k,v in sorted(sessions.items(),key=lambda x:x[0],reverse=True)]

    @fa.post("/result/{sid}/retune")
    async def retune(sid:str, req:RTR):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        tn=req.tuning
        if tn not in TUNINGS: raise HTTPException(400,f"Unknown: {tn}")
        sd=Path(s["session_dir"])
        op=sd/"notes_assigned_original.json"; ap_=sd/"notes_assigned.json"
        if not op.exists() and ap_.exists(): shutil.copy2(ap_,op)
        np_=op if op.exists() else ap_
        if not np_.exists(): np_=sd/"notes.json"
        if not np_.exists(): raise HTTPException(404,"Notes not found")
        nd=json.loads(np_.read_text(encoding="utf-8"))
        notes_=copy.deepcopy(nd if isinstance(nd,list) else nd.get("notes",nd))
        tu=TUNINGS[tn]; capo=req.capo if req.capo is not None else s.get("capo",0)
        if capo is None: capo=0
        ct=[p+capo for p in tu] if capo>0 else tu
        from string_assigner import assign_strings_dp
        from tab_renderer import notes_to_tab_musicxml
        notes_=assign_strings_dp(notes_, tuning=ct)
        ap_.write_text(json.dumps(notes_,ensure_ascii=False,indent=2),encoding="utf-8")
        bp=sd/"beats.json"; beats_=[]; bpm_r=s.get("bpm",120); ts_=s.get("time_signature","4/4")
        if bp.exists():
            bd=json.loads(bp.read_text(encoding="utf-8")); beats_=bd.get("beats",[]); bpm_r=bd.get("bpm",bpm_r); ts_=bd.get("time_signature",ts_)
        xc,_=notes_to_tab_musicxml(notes_,beats=beats_,bpm=bpm_r,title=s.get("filename",sid),tuning=ct,time_signature=ts_,noise_gate=req.noise_gate)
        (sd/"tab.musicxml").write_text(xc,encoding="utf-8")
        s["tuning"]=tn; s["capo"]=capo; s["total_notes"]=len(notes_); save(sid)
        return {"status":"ok","tuning":tn,"capo":capo,"total_notes":len(notes_)}

    @fa.get("/health")
    async def health():
        model_dir = "/app/music-transcription/python/_processed_guitarset_data/training_output"
        models_found = []
        for e in _EXPERTS:
            p = os.path.join(model_dir, e, "best_model.pth")
            models_found.append({"name":e,"exists":os.path.exists(p),"size_mb":round(os.path.getsize(p)/1024/1024,1) if os.path.exists(p) else 0})
        vol_sessions = len(list(UPLOAD_DIR.iterdir())) if UPLOAD_DIR.exists() else 0
        return {"status":"healthy","app":"SoloTab (Modal)","version":"2.0.0","models":models_found,
                "sessions_on_volume": vol_sessions, "sessions_in_memory": len(sessions)}

    return fa
