"""
SoloTab V2.2 — Modal デプロイメント
====================================
フルパイプライン: librosa ビート検出 + Pure MoE 推論 + テクニック検出 + MusicXML 生成

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
    "finetuned_martin_finger_multitask_3ds_ga",
    "finetuned_taylor_finger_multitask_3ds_ga",
    "finetuned_luthier_finger_multitask_3ds_ga",
    "finetuned_martin_pick_multitask_3ds_ga",
    "finetuned_taylor_pick_multitask_3ds_ga",
    "finetuned_luthier_pick_multitask_3ds_ga",
    "finetuned_gibson_thumb_multitask_3ds_ga",
]

solotab_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "fonts-noto-cjk")
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
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
        "basic-pitch",
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
    gpu="T4",
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

    from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Response, Request
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

    def save(sid, commit=True):
        if sid in sessions:
            sd = Path(sessions[sid]["session_dir"])
            (sd/"session.json").write_text(json.dumps(sessions[sid], ensure_ascii=False, indent=2), encoding="utf-8")
            if commit:
                try:
                    session_vol.commit()
                except Exception as e:
                    print(f"[save] Volume commit warning: {e}")

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
        """BP+MoE融合パイプライン (BasicPitch高Recall + MoE弦情報 + Viterbi DP + テクニック検出)"""
        s = sessions[sid]
        sd = Path(s["session_dir"])
        wav = Path(s["wav_path"])
        try:
            s["status"] = SS.PROCESSING; save(sid, commit=False)

            # === Step 0: ビート検出 ===
            s["progress"] = "ビート検出中..."; s["steps_done"] = 0; save(sid, commit=False)
            from beat_detector import detect_beats
            beat_result = detect_beats(str(wav))
            beats_list = beat_result["beats"]
            bpm = beat_result["bpm"]
            time_sig = beat_result.get("time_signature", "4/4")
            (sd/"beats.json").write_text(json.dumps(beat_result, ensure_ascii=False), encoding="utf-8")

            # === Step 1: MoE推論 + BasicPitch推論 + 融合 ===
            s["progress"] = "ノート検出中 (MoE+BP)..."; s["steps_done"] = 1; save(sid, commit=False)

            # MoE推論
            from pure_moe_transcriber import transcribe_pure_moe
            moe_notes = transcribe_pure_moe(str(wav), vote_threshold=2, onset_threshold=0.3, vote_prob_threshold=0.3)
            print(f"MoE: {len(moe_notes)} notes")

            # BasicPitch推論
            bp_notes = []
            try:
                from basic_pitch.inference import predict as bp_predict
                _, midi_data, _ = bp_predict(str(wav))
                for inst in midi_data.instruments:
                    for note in inst.notes:
                        bp_notes.append({
                            "start": float(note.start),
                            "end": float(note.end),
                            "pitch": int(note.pitch),
                            "velocity": round(note.velocity / 127.0, 4),
                        })
                print(f"BasicPitch: {len(bp_notes)} notes")
            except Exception as e:
                print(f"BasicPitch failed, MoE only: {e}")

            # 融合: BP∩MoE + MoE高確信 + BP独自
            MATCH_ONSET_TOL = 0.10
            MATCH_PITCH_TOL = 1

            if bp_notes and moe_notes:
                fused = []
                used_moe = set()
                used_bp = set()

                # 1. BP∩MoE: 両方が検出（MoEの弦情報を使用）
                for i, bp_n in enumerate(bp_notes):
                    for j, moe_n in enumerate(moe_notes):
                        if j in used_moe:
                            continue
                        if (abs(bp_n["start"] - moe_n["start"]) < MATCH_ONSET_TOL
                                and abs(bp_n["pitch"] - moe_n["pitch"]) <= MATCH_PITCH_TOL):
                            fused.append(moe_n)
                            used_moe.add(j)
                            used_bp.add(i)
                            break

                # 2. MoE独自: 高確信ノート
                for j, moe_n in enumerate(moe_notes):
                    if j not in used_moe:
                        if float(moe_n.get("velocity", 0)) >= 0.65:
                            fused.append(moe_n)

                # 3. BP独自: MoEが拾えなかった音（ベース音・和音補完）
                #    確信度フィルタ: velocity低いBP独自ノートは除外（音が多すぎ防止）
                for i, bp_n in enumerate(bp_notes):
                    if i not in used_bp:
                        if float(bp_n.get("velocity", 0)) >= 0.70:
                            fused.append(bp_n)

                notes = sorted(fused, key=lambda n: (n["start"], n["pitch"]))
                n_matched = len(used_moe)
                n_moe_solo = sum(1 for j in range(len(moe_notes)) if j not in used_moe and float(moe_notes[j].get("velocity",0))>=0.75)
                n_bp_solo = len(bp_notes) - len(used_bp)
                print(f"融合: BP∩MoE={n_matched}, MoE独自={n_moe_solo}, BP独自={n_bp_solo} → total={len(notes)}")
            elif moe_notes:
                notes = moe_notes
            elif bp_notes:
                notes = bp_notes
            else:
                notes = []

            # === Step 2: Viterbi DP 弦・フレット最適化 + テクニック検出 ===
            s["progress"] = "弦・フレット最適化中..."; s["steps_done"] = 2; save(sid, commit=False)
            try:
                from string_assigner import assign_strings_dp
                tuning = TUNINGS.get(s.get("tuning", "standard"), TUNINGS["standard"])
                notes = assign_strings_dp(notes, tuning=tuning)
                print(f"Viterbi DP: {len(notes)} notes assigned")
            except Exception as e:
                print(f"String assignment failed (using raw): {e}")
                import traceback; traceback.print_exc()

            # テクニック検出（Viterbi DP後の正確な弦割り当てで実行）
            # 先に調号を推定してテクニック判定に使用
            _key_sig = "C"
            try:
                from music_theory import detect_key_signature
                _key_sig = detect_key_signature(notes)
            except Exception:
                pass

            try:
                from technique_detector import detect_techniques, add_techniques_to_musicxml_notes
                notes = detect_techniques(notes, bpm=bpm, key_signature=_key_sig)
                notes = add_techniques_to_musicxml_notes(notes)
                tech_count = sum(1 for n in notes if n.get("technique") and n["technique"] != "normal")
                tech_types = {}
                for n in notes:
                    t = n.get("technique", "normal")
                    tech_types[t] = tech_types.get(t, 0) + 1
                print(f"Techniques detected: {tech_count} — {tech_types}")
            except Exception as e:
                print(f"Technique detection skipped: {e}")

            (sd/"notes_assigned.json").write_text(json.dumps(_to_native(notes), ensure_ascii=False, indent=2), encoding="utf-8")

            # === Step 3: 音楽理論解析 + TAB譜生成 ===
            s["progress"] = "音楽理論解析中..."; s["steps_done"] = 3
            from tab_renderer import notes_to_tab_musicxml
            from music_theory import apply_music_theory, detect_rhythm_pattern, detect_key_signature, assign_dynamics_to_bars
            tuning = TUNINGS.get(s.get("tuning","standard"), TUNINGS["standard"])
            title = s.get("filename", sid)
            ext = os.path.splitext(title)[1]
            if ext.lower() in {".mp3",".wav",".m4a",".flac",".ogg"}: title = title[:-len(ext)]

            # --- 音楽理論解析 ---
            try:
                rhythm_info = detect_rhythm_pattern(notes, beats_list)
                key_sig = detect_key_signature(notes)
                print(f"Music theory: rhythm={rhythm_info['subdivision']} "
                      f"(triplet_ratio={rhythm_info['triplet_ratio']:.2f}), "
                      f"key={key_sig}")
            except Exception as e:
                print(f"Music theory analysis failed: {e}")
                rhythm_info = {"subdivision": "straight", "triplet_ratio": 0.0}
                key_sig = "C"

            xml_content, tech_map = notes_to_tab_musicxml(
                notes, beats=beats_list, bpm=bpm, title=title,
                tuning=tuning, time_signature=time_sig,
                rhythm_info=rhythm_info, key_signature=key_sig
            )
            (sd/"tab.musicxml").write_text(xml_content, encoding="utf-8")
            (sd/"techniques.json").write_text(json.dumps(_to_native(tech_map)), encoding="utf-8")
            print(f"MusicXML written: {len(xml_content)} chars")

            # ファイルをVolume上で可視化（ステータス変更前にcommit）
            try:
                session_vol.commit()
            except Exception:
                pass
            print(f"Session {sid} files committed")

            # ステータスを最後にcompletedに変更（SSEがこれを検知した時点でファイルは既に可視）
            s.update({"status":SS.COMPLETED,"total_notes":len(notes),"bpm":bpm,
                      "time_signature":time_sig,"progress":"解析完了","steps_done":4})
            save(sid)
            print(f"Session {sid} completed")
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
        # ファイル存在チェックでステータスを自動補正
        sd = Path(s["session_dir"])
        if s.get("status") == SS.PROCESSING:
            if (sd/"tab.musicxml").exists():
                notes_file = sd/"notes_assigned.json"
                n_notes = len(json.loads(notes_file.read_text(encoding="utf-8"))) if notes_file.exists() else 0
                s.update({"status":SS.COMPLETED, "progress":"解析完了", "steps_done":4, "total_notes":n_notes})
                sessions[sid] = s
            elif (sd/"notes_assigned.json").exists():
                s["progress"] = "TAB譜生成中..."; s["steps_done"] = 3
            elif (sd/"beats.json").exists():
                s["progress"] = "MoE推論中..."; s["steps_done"] = 1
        return SR(session_id=sid,status=s["status"],progress=s.get("progress"),error=s.get("error"),filename=s.get("filename"))

    @fa.get("/result/{sid}",response_model=RR)
    async def result(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        # ファイル存在チェックで自動補正
        sd = Path(s["session_dir"])
        if s["status"]!=SS.COMPLETED:
            if not (sd/"tab.musicxml").exists():
                try: session_vol.reload()
                except Exception: pass
            if (sd/"tab.musicxml").exists():
                notes_file = sd/"notes_assigned.json"
                n_notes = len(json.loads(notes_file.read_text(encoding="utf-8"))) if notes_file.exists() else 0
                beats_file = sd/"beats.json"
                bpm = 120
                if beats_file.exists():
                    bd = json.loads(beats_file.read_text(encoding="utf-8")); bpm = bd.get("bpm", 120)
                s.update({"status":SS.COMPLETED, "total_notes":n_notes, "bpm":bpm})
                sessions[sid] = s
        if s["status"]!=SS.COMPLETED: raise HTTPException(202,"Not complete")
        return RR(session_id=sid,status=s["status"],bpm=s.get("bpm"),filename=s.get("filename"),total_notes=s.get("total_notes"),tuning=s.get("tuning"),key=s.get("key"),capo=s.get("capo"),suggested_tuning=s.get("suggested_tuning"))

    @fa.get("/result/{sid}/musicxml")
    async def musicxml(sid:str):
        s = load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        xp=Path(s["session_dir"])/"tab.musicxml"
        if not xp.exists():
            # Volume同期: 別コンテナでcommitされた可能性
            try: session_vol.reload()
            except Exception: pass
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

    @fa.patch("/result/{sid}/notes/{note_idx}")
    async def edit_note(sid: str, note_idx: int, body: dict):
        """ノートの編集（フレット変更 or 削除）→ MusicXML再生成"""
        s = load_session(sid)
        if s is None: raise HTTPException(404, "Not found")
        sd = Path(s["session_dir"])
        np_ = sd / "notes_assigned.json"
        if not np_.exists(): raise HTTPException(404, "Notes not found")

        notes_data = json.loads(np_.read_text(encoding="utf-8"))

        if body.get("delete"):
            # ノート削除
            if 0 <= note_idx < len(notes_data):
                notes_data.pop(note_idx)
            else:
                raise HTTPException(400, f"Invalid note index: {note_idx}")
        else:
            # フレット・弦変更
            new_fret = body.get("fret")
            new_string = body.get("string")
            if new_fret is None:
                raise HTTPException(400, "fret is required")
            new_fret = int(new_fret)
            if not (0 <= new_fret <= 15):
                raise HTTPException(400, f"Invalid fret: {new_fret}")
            if new_string is not None:
                new_string = int(new_string)
                if not (1 <= new_string <= 6):
                    raise HTTPException(400, f"Invalid string: {new_string}")
            if 0 <= note_idx < len(notes_data):
                note = notes_data[note_idx]
                # 弦変更
                if new_string is not None:
                    note["string"] = new_string
                string = note.get("string", 1)
                # ピッチも更新（開放弦ピッチ + 新フレット）
                tuning = TUNINGS.get(s.get("tuning", "standard"), TUNINGS["standard"])
                string_idx = 6 - string  # MusicXML弦番号 → 内部インデックス
                if 0 <= string_idx < len(tuning):
                    note["pitch"] = tuning[string_idx] + new_fret
                note["fret"] = new_fret
            else:
                raise HTTPException(400, f"Invalid note index: {note_idx}")

        # 保存
        np_.write_text(json.dumps(notes_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # MusicXML再生成
        try:
            from tab_renderer import notes_to_tab_musicxml
            tuning = TUNINGS.get(s.get("tuning", "standard"), TUNINGS["standard"])
            bp = sd / "beats.json"
            beats_ = []; bpm_r = s.get("bpm", 120); ts_ = s.get("time_signature", "4/4")
            if bp.exists():
                bd = json.loads(bp.read_text(encoding="utf-8"))
                beats_ = bd.get("beats", []); bpm_r = bd.get("bpm", bpm_r); ts_ = bd.get("time_signature", ts_)
            title = s.get("filename", sid).rsplit(".", 1)[0]
            xc, _ = notes_to_tab_musicxml(
                notes_data, beats=beats_, bpm=bpm_r, title=title,
                tuning=tuning, time_signature=ts_
            )
            (sd / "tab.musicxml").write_text(xc, encoding="utf-8")
        except Exception as e:
            print(f"MusicXML regeneration failed: {e}")

        s["total_notes"] = len(notes_data)
        save(sid)
        return {"status": "ok", "total_notes": len(notes_data)}

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
        tu=TUNINGS[tn]; capo=req.capo if req.capo is not None else s.get("capo",0)
        if capo is None: capo=0
        ct=[p+capo for p in tu] if capo>0 else tu

        # チューニング・カポが変更されたかチェック
        tuning_changed = (tn != s.get("tuning","standard")) or (capo != s.get("capo",0))

        ap_=sd/"notes_assigned.json"

        if tuning_changed:
            # チューニング/カポ変更 → 弦再割り当てが必要
            op=sd/"notes_assigned_original.json"
            if not op.exists() and ap_.exists(): shutil.copy2(ap_,op)
            np_=op if op.exists() else ap_
            if not np_.exists(): np_=sd/"notes.json"
            if not np_.exists(): raise HTTPException(404,"Notes not found")
            nd=json.loads(np_.read_text(encoding="utf-8"))
            notes_=copy.deepcopy(nd if isinstance(nd,list) else nd.get("notes",nd))
            from string_assigner import assign_strings_dp
            notes_=assign_strings_dp(notes_, tuning=ct)
            ap_.write_text(json.dumps(notes_,ensure_ascii=False,indent=2),encoding="utf-8")
        else:
            # CUTのみ変更 → 既存のnotes_assigned.jsonをそのまま使用
            if not ap_.exists():
                raise HTTPException(404,"Notes not found")
            notes_=json.loads(ap_.read_text(encoding="utf-8"))
            if not isinstance(notes_, list):
                notes_ = notes_.get("notes", notes_)

        from tab_renderer import notes_to_tab_musicxml
        bp=sd/"beats.json"; beats_=[]; bpm_r=s.get("bpm",120); ts_=s.get("time_signature","4/4")
        if bp.exists():
            bd=json.loads(bp.read_text(encoding="utf-8")); beats_=bd.get("beats",[]); bpm_r=bd.get("bpm",bpm_r); ts_=bd.get("time_signature",ts_)
        xc,_=notes_to_tab_musicxml(notes_,beats=beats_,bpm=bpm_r,title=s.get("filename",sid),tuning=ct,time_signature=ts_,noise_gate=req.noise_gate)
        (sd/"tab.musicxml").write_text(xc,encoding="utf-8")
        s["tuning"]=tn; s["capo"]=capo; s["total_notes"]=len(notes_); save(sid)
        return {"status":"ok","tuning":tn,"capo":capo,"total_notes":len(notes_)}

    # ─── Score Model API (PowerTab互換) ───

    @fa.get("/result/{sid}/score")
    async def get_score(sid:str):
        """score.jsonを返す。存在しなければnotes_assignedから自動生成"""
        s=load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        sd=Path(s["session_dir"])
        score_path=sd/"score.json"
        if score_path.exists():
            return json.loads(score_path.read_text(encoding="utf-8"))
        # マイグレーション
        na=sd/"notes_assigned.json"
        if not na.exists(): raise HTTPException(404,"No data")
        notes=json.loads(na.read_text(encoding="utf-8"))
        beats_file=sd/"beats.json"
        beats_list=[]
        bpm_v=s.get("bpm",120)
        ts_v=s.get("time_signature","4/4")
        if beats_file.exists():
            bd=json.loads(beats_file.read_text(encoding="utf-8"))
            beats_list=bd.get("beats",[]); bpm_v=bd.get("bpm",bpm_v); ts_v=bd.get("time_signature",ts_v)
        from score_model import migrate_from_notes_assigned
        score=migrate_from_notes_assigned(notes,beats_list,bpm=bpm_v,
            title=s.get("filename","").rsplit(".",1)[0],tuning=s.get("tuning","standard"),time_signature=ts_v)
        score_path.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
        session_vol.commit()
        return score

    @fa.put("/result/{sid}/score")
    async def put_score(sid:str, request:Request):
        """score.json全体を保存し、MusicXMLを再生成"""
        s=load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        sd=Path(s["session_dir"])
        score=await request.json()
        sd_path=sd/"score.json"
        sd_path.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
        # MusicXML再生成
        from score_model import score_to_musicxml
        ct=TUNINGS.get(score.get("meta",{}).get("tuning","standard"),TUNINGS["standard"])
        xml=score_to_musicxml(score,tuning_midi=ct)
        (sd/"tab.musicxml").write_text(xml,encoding="utf-8")
        s["total_notes"]=sum(len(b.get("notes",[]))for bar in score.get("bars",[])for b in bar.get("beats",[]))
        save(sid)
        return {"status":"ok","total_notes":s["total_notes"]}

    @fa.post("/result/{sid}/score/bars")
    async def add_bar(sid:str, request:Request):
        """小節を追加"""
        s=load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        sd=Path(s["session_dir"])
        body=await request.json()
        after_bar=body.get("after_bar",None)  # Noneなら末尾
        score_path=sd/"score.json"
        if not score_path.exists(): raise HTTPException(404,"score.json not found")
        score=json.loads(score_path.read_text(encoding="utf-8"))
        from score_model import empty_bar
        ts=score.get("meta",{}).get("time_signature","4/4")
        bpm=score.get("meta",{}).get("bpm",120)
        new_bar=empty_bar(bar_number=len(score["bars"])+1,time_signature=ts,tempo=bpm)
        if after_bar is not None and 0<=after_bar<len(score["bars"]):
            score["bars"].insert(after_bar+1,new_bar)
        else:
            score["bars"].append(new_bar)
        # 番号振り直し
        for i,b in enumerate(score["bars"]): b["bar_number"]=i+1
        score_path.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
        from score_model import score_to_musicxml
        ct=TUNINGS.get(score.get("meta",{}).get("tuning","standard"),TUNINGS["standard"])
        xml=score_to_musicxml(score,tuning_midi=ct)
        (sd/"tab.musicxml").write_text(xml,encoding="utf-8"); save(sid)
        return {"status":"ok","total_bars":len(score["bars"])}

    @fa.delete("/result/{sid}/score/bars/{bar_idx}")
    async def delete_bar(sid:str, bar_idx:int):
        """小節を削除"""
        s=load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        sd=Path(s["session_dir"])
        score_path=sd/"score.json"
        if not score_path.exists(): raise HTTPException(404,"score.json not found")
        score=json.loads(score_path.read_text(encoding="utf-8"))
        if bar_idx<0 or bar_idx>=len(score["bars"]): raise HTTPException(400,"Invalid bar index")
        if len(score["bars"])<=1: raise HTTPException(400,"Cannot delete last bar")
        score["bars"].pop(bar_idx)
        for i,b in enumerate(score["bars"]): b["bar_number"]=i+1
        score_path.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
        from score_model import score_to_musicxml
        ct=TUNINGS.get(score.get("meta",{}).get("tuning","standard"),TUNINGS["standard"])
        xml=score_to_musicxml(score,tuning_midi=ct)
        (sd/"tab.musicxml").write_text(xml,encoding="utf-8"); save(sid)
        return {"status":"ok","total_bars":len(score["bars"])}

    @fa.patch("/result/{sid}/score/bars/{bar_idx}")
    async def patch_bar(sid:str, bar_idx:int, request:Request):
        """小節プロパティを更新(拍子/テンポ/リハーサル/コード/ダイナミクス/小節線/テキスト)"""
        s=load_session(sid)
        if s is None: raise HTTPException(404,"Not found")
        sd=Path(s["session_dir"])
        score_path=sd/"score.json"
        if not score_path.exists(): raise HTTPException(404,"score.json not found")
        score=json.loads(score_path.read_text(encoding="utf-8"))
        if bar_idx<0 or bar_idx>=len(score["bars"]): raise HTTPException(400,"Invalid bar index")
        body=await request.json()
        bar=score["bars"][bar_idx]
        for k in ["time_signature","tempo","rehearsal_sign","chord_text","dynamic",
                   "barline_start","barline_end","direction","alternate_ending","text_items","key_signature"]:
            if k in body: bar[k]=body[k]
        score_path.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
        from score_model import score_to_musicxml
        ct=TUNINGS.get(score.get("meta",{}).get("tuning","standard"),TUNINGS["standard"])
        xml=score_to_musicxml(score,tuning_midi=ct)
        (sd/"tab.musicxml").write_text(xml,encoding="utf-8"); save(sid)
        return {"status":"ok","bar":bar}

    def _regen_xml(sid, score, sd):
        """共通: score→MusicXML再生成+保存"""
        from score_model import score_to_musicxml
        s=sessions.get(sid,{})
        ct=TUNINGS.get(score.get("meta",{}).get("tuning","standard"),TUNINGS["standard"])
        xml=score_to_musicxml(score,tuning_midi=ct)
        (sd/"tab.musicxml").write_text(xml,encoding="utf-8")
        (sd/"score.json").write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
        save(sid)

    @fa.post("/result/{sid}/score/bars/{bar_idx}/beats")
    async def add_beat(sid:str, bar_idx:int, request:Request):
        """ビート(ノートグループ)を追加"""
        s=load_session(sid)
        if s is None: raise HTTPException(404)
        sd=Path(s["session_dir"]); score=json.loads((sd/"score.json").read_text(encoding="utf-8"))
        if bar_idx<0 or bar_idx>=len(score["bars"]): raise HTTPException(400)
        body=await request.json()
        from score_model import empty_beat, empty_note
        position=body.get("position",0)
        duration=body.get("duration","quarter")
        fret=body.get("fret",0); string=body.get("string",1); pitch=body.get("pitch",40)
        beat=empty_beat(position=position, duration=duration)
        if not body.get("rest",False):
            beat["notes"]=[empty_note(string=string,fret=fret,pitch=pitch)]
        else:
            beat["rest"]=True
        score["bars"][bar_idx]["beats"].append(beat)
        score["bars"][bar_idx]["beats"].sort(key=lambda b:b.get("position",0))
        _regen_xml(sid,score,sd)
        return {"status":"ok"}

    @fa.delete("/result/{sid}/score/bars/{bar_idx}/beats/{beat_idx}")
    async def delete_beat(sid:str, bar_idx:int, beat_idx:int):
        """ビートを削除"""
        s=load_session(sid)
        if s is None: raise HTTPException(404)
        sd=Path(s["session_dir"]); score=json.loads((sd/"score.json").read_text(encoding="utf-8"))
        if bar_idx<0 or bar_idx>=len(score["bars"]): raise HTTPException(400)
        beats=score["bars"][bar_idx]["beats"]
        if beat_idx<0 or beat_idx>=len(beats): raise HTTPException(400)
        beats.pop(beat_idx)
        _regen_xml(sid,score,sd)
        return {"status":"ok"}

    @fa.patch("/result/{sid}/score/bars/{bar_idx}/beats/{beat_idx}")
    async def patch_beat(sid:str, bar_idx:int, beat_idx:int, request:Request):
        """ビートプロパティ変更(音価/休符/3連符/タイ)"""
        s=load_session(sid)
        if s is None: raise HTTPException(404)
        sd=Path(s["session_dir"]); score=json.loads((sd/"score.json").read_text(encoding="utf-8"))
        if bar_idx<0 or bar_idx>=len(score["bars"]): raise HTTPException(400)
        beats=score["bars"][bar_idx]["beats"]
        if beat_idx<0 or beat_idx>=len(beats): raise HTTPException(400)
        body=await request.json()
        beat=beats[beat_idx]
        for k in ["duration","dotted","double_dotted","rest","tie","triplet","position"]:
            if k in body: beat[k]=body[k]
        _regen_xml(sid,score,sd)
        return {"status":"ok","beat":beat}

    @fa.post("/result/{sid}/score/bars/{bar_idx}/beats/{beat_idx}/notes")
    async def add_note(sid:str, bar_idx:int, beat_idx:int, request:Request):
        """ビートにノートを追加(和音)"""
        s=load_session(sid)
        if s is None: raise HTTPException(404)
        sd=Path(s["session_dir"]); score=json.loads((sd/"score.json").read_text(encoding="utf-8"))
        bar=score["bars"][bar_idx]; beat=bar["beats"][beat_idx]
        body=await request.json()
        from score_model import empty_note
        n=empty_note(string=body.get("string",1),fret=body.get("fret",0),
                     pitch=body.get("pitch",40),velocity=body.get("velocity",0.8))
        # テクニック指定
        if "techniques" in body:
            for k,v in body["techniques"].items():
                if k in n["techniques"]: n["techniques"][k]=v
        beat["notes"].append(n)
        beat["rest"]=False
        _regen_xml(sid,score,sd)
        return {"status":"ok"}

    @fa.delete("/result/{sid}/score/bars/{bar_idx}/beats/{beat_idx}/notes/{note_idx}")
    async def delete_note(sid:str, bar_idx:int, beat_idx:int, note_idx:int):
        """ノートを削除"""
        s=load_session(sid)
        if s is None: raise HTTPException(404)
        sd=Path(s["session_dir"]); score=json.loads((sd/"score.json").read_text(encoding="utf-8"))
        notes=score["bars"][bar_idx]["beats"][beat_idx]["notes"]
        if note_idx<0 or note_idx>=len(notes): raise HTTPException(400)
        notes.pop(note_idx)
        if not notes: score["bars"][bar_idx]["beats"][beat_idx]["rest"]=True
        _regen_xml(sid,score,sd)
        return {"status":"ok"}

    @fa.patch("/result/{sid}/score/bars/{bar_idx}/beats/{beat_idx}/notes/{note_idx}")
    async def patch_note(sid:str, bar_idx:int, beat_idx:int, note_idx:int, request:Request):
        """ノートプロパティ変更(フレット/弦/テクニック) + ピッチ⇔フレット自動同期"""
        s=load_session(sid)
        if s is None: raise HTTPException(404)
        sd=Path(s["session_dir"]); score=json.loads((sd/"score.json").read_text(encoding="utf-8"))
        note=score["bars"][bar_idx]["beats"][beat_idx]["notes"][note_idx]
        body=await request.json()

        from score_model import fret_string_to_pitch, pitch_to_best_fret_string, note_name_to_pitch
        ct=TUNINGS.get(score.get("meta",{}).get("tuning","standard"),TUNINGS["standard"])

        # 音名文字列でのピッチ指定 (例: "C4", "D#3")
        if "note_name" in body:
            p = note_name_to_pitch(body["note_name"])
            if p is not None: body["pitch"] = p

        has_fret = "fret" in body
        has_string = "string" in body
        has_pitch = "pitch" in body

        # 基本プロパティ反映
        for k in ["string","fret","pitch","velocity"]:
            if k in body: note[k]=body[k]

        # 自動同期
        if (has_fret or has_string) and not has_pitch:
            # TAB変更 → ピッチ自動計算
            note["pitch"] = fret_string_to_pitch(note["string"], note["fret"], ct)
        elif has_pitch and not has_fret and not has_string:
            # 五線譜変更 → 最適フレット/弦自動計算
            s_num, f_num = pitch_to_best_fret_string(note["pitch"], ct, prefer_string=note.get("string"))
            note["string"] = s_num
            note["fret"] = f_num

        if "techniques" in body:
            for k,v in body["techniques"].items():
                if k in note["techniques"]: note["techniques"][k]=v
        _regen_xml(sid,score,sd)
        return {"status":"ok","note":note}

    @fa.get("/health")
    async def health():
        model_dir = "/app/music-transcription/python/_processed_guitarset_data/training_output"
        models_found = []
        for e in _EXPERTS:
            p = os.path.join(model_dir, e, "best_model.pth")
            models_found.append({"name":e,"exists":os.path.exists(p),"size_mb":round(os.path.getsize(p)/1024/1024,1) if os.path.exists(p) else 0})
        vol_sessions = len(list(UPLOAD_DIR.iterdir())) if UPLOAD_DIR.exists() else 0
        return {"status":"healthy","app":"SoloTab (Modal)","version":"2.3.0","models":models_found,
                "sessions_on_volume": vol_sessions, "sessions_in_memory": len(sessions)}

    return fa
