"""
backend/benchmark/test_real_world_sakura.py
============================================
ユーザー実楽曲（sakurasakukoro.mp3）を用いた真のE2Eテストハーネス。
本番と同一の pipeline.py 全工程（AMT、ビート、キー、コード、チューニング、GP5生成）を実行し、
ユーザー体験（ノート数・チューニング・コード幻出・同期）を冷酷に判定する。
"""

import sys
import os
import shutil
import json
import uuid
from pathlib import Path
import guitarpro as gp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import run_pipeline

def run_real_world_e2e_test():
    print("=== TASK-928: 真のE2Eテスト実行（sakurasakukoro.mp3） ===", flush=True)
    
    src_wav = Path("uploads/20260816-140733-0402e3/converted.wav")
    if not src_wav.exists():
        print(f"[FAIL] 音源ファイルが見つかりません: {src_wav}")
        return False
        
    sid_hex = uuid.uuid4().hex[:6]
    session_id = f"e2e_test_{sid_hex}"
    session_dir = Path("uploads") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_wav, session_dir / "converted.wav")
    
    print(f"[E2E] セッション開始: {session_id}, tuning_name='standard' (ユーザー指定)")
    res = run_pipeline(
        session_id=session_id,
        session_dir=session_dir,
        wav_path=session_dir / "converted.wav",
        tuning_name="standard",
        skip_demucs=True,
        fast_moe=True
    )
    
    gp5_path = session_dir / "tab.gp5"
    if not gp5_path.exists():
        print(f"[FAIL] GP5ファイルが生成されませんでした: {gp5_path}")
        return False
        
    song = gp.parse(str(gp5_path))
    track1 = song.tracks[0]
    
    # 1. チューニング検証
    gp5_tuning = [s.value for s in track1.strings]
    expected_tuning = [64, 59, 55, 50, 45, 40]
    tuning_pass = (gp5_tuning == expected_tuning) and (res.get("tuning") == "standard")
    
    # 2. 総ノート数検証
    total_notes = 0
    open_notes = 0
    for m in track1.measures:
        for v in m.voices:
            for b in v.beats:
                for n in b.notes:
                    total_notes += 1
                    if n.value == 0:
                        open_notes += 1
    notes_pass = total_notes >= 700
    
    # 3. コード検出の無音区間検証
    with open(session_dir / "chords.json", encoding="utf-8") as f:
        chords = json.load(f)
    
    bar1_chords = [c for c in chords if c["start"] < 8.0]
    bar1_silent_pass = any(c.get("chord") == "N.C." for c in bar1_chords if c["start"] < 4.5)
    
    # 4. 音声全長の小節グリッド網羅性検証
    with open(session_dir / "beats.json", encoding="utf-8") as f:
        b_data = json.load(f)
    beats = b_data["beats"]
    gp5_measures_count = len(track1.measures)
    expected_measures = int(len(beats) / 4)
    duration_coverage_pass = gp5_measures_count >= expected_measures
    
    print("\n--- E2E テスト判定結果 ---")
    print(f"1. チューニングSSOT: {'PASS' if tuning_pass else 'FAIL'} | GP5={gp5_tuning} vs Expected={expected_tuning}")
    print(f"2. ノート数・ポリフォニー: {'PASS' if notes_pass else 'FAIL'} | 総ノート={total_notes} (基準 >= 700), 開放弦={open_notes}")
    print(f"3. 無音小節コード幻出遮断: {'PASS' if bar1_silent_pass else 'FAIL'} | Bar1 Chords={[c['chord'] for c in bar1_chords]}")
    print(f"4. 音声全長小節グリッド網羅: {'PASS' if duration_coverage_pass else 'FAIL'} | 小節数={gp5_measures_count} vs 期待小節数={expected_measures}")
    
    all_pass = tuning_pass and notes_pass and bar1_silent_pass and duration_coverage_pass
    print(f"\n総合判定: {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass

if __name__ == "__main__":
    run_real_world_e2e_test()
