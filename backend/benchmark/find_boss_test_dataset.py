"""
datasets/solotab26k/boss_test/ へのラスボスGP5抽出スクリプト
"""

import os
import shutil
from pathlib import Path
import guitarpro

SEARCH_DIRS = [
    Path("D:/Music/chordlink-solotab/gprotab_downloads"),
    Path("D:/Music/chordlink-solotab/datasets"),
    Path("D:/Music/datasets"),
    Path("C:/Users/kotan/Downloads"),
    Path("C:/Users/kotan/Desktop"),
]

DEST_DIR = Path("D:/Music/chordlink-solotab/datasets/solotab26k/boss_test")
DEST_DIR.mkdir(parents=True, exist_ok=True)

TARGET_KEYWORDS = ["kotaro", "oshio", "andy", "mckee", "tommy", "emmanuel", "pressplay", "sungha", "jung", "drifting", "rylynn", "tight", "wings", "wind", "fight", "splash"]

SPECIAL_TUNINGS = {
    (38, 45, 50, 55, 57, 62): "DADGAD",
    (38, 45, 50, 54, 57, 62): "Open D",
    (38, 43, 50, 55, 59, 62): "Open G",
    (36, 43, 48, 53, 57, 62): "Drop C",
    (36, 43, 48, 55, 60, 64): "Open C",
}

found_files = []
seen_hashes = set()

print("=== ラスボス（Boss Test）ファイルの全走査開始 ===")

for d in SEARCH_DIRS:
    if not d.exists():
        continue
    for p in d.rglob("*.gp*"):
        if p.suffix.lower() not in (".gp3", ".gp4", ".gp5"):
            continue
        
        lower_name = p.name.lower()
        matched = any(kw in lower_name for kw in TARGET_KEYWORDS)
        
        try:
            song = guitarpro.parse(str(p))
            artist_meta = (song.artist or "").lower()
            title_meta = (song.title or "").lower()
            
            if not matched:
                matched = any(kw in artist_meta or kw in title_meta for kw in TARGET_KEYWORDS)
            
            # ギタートラック検査
            for track in song.tracks:
                if track.isPercussionTrack or len(track.strings) != 6:
                    continue
                
                tuning = tuple(s.value for s in track.strings)
                tuning_name = SPECIAL_TUNINGS.get(tuning, "Standard" if tuning == (40, 45, 50, 55, 59, 64) else "Other")
                
                # 特殊奏法カウント
                total_notes = 0
                tech_notes = 0
                
                for measure in track.measures:
                    for voice in measure.voices:
                        for beat in voice.beats:
                            for note in beat.notes:
                                total_notes += 1
                                if note.effect.harmonic is not None:
                                    tech_notes += 1
                                elif getattr(note.effect, "isTapping", False) or getattr(note.effect, "isSlapping", False):
                                    tech_notes += 1
                                elif note.effect.hammer or note.effect.slides:
                                    tech_notes += 0.5
                
                tech_ratio = (tech_notes / total_notes) if total_notes > 0 else 0.0
                
                # 採用判定
                is_boss = False
                if matched and total_notes >= 30:
                    is_boss = True
                elif tech_ratio >= 0.05 and total_notes >= 40:
                    is_boss = True
                elif tuning_name in SPECIAL_TUNINGS.values() and total_notes >= 40:
                    is_boss = True
                
                if is_boss:
                    file_key = f"{song.artist}_{song.title}_{total_notes}"
                    if file_key in seen_hashes:
                        break
                    seen_hashes.add(file_key)
                    
                    dest_file = DEST_DIR / p.name
                    if not dest_file.exists():
                        shutil.copy2(str(p), str(dest_file))
                    
                    found_files.append({
                        "filename": p.name,
                        "artist": song.artist or "Unknown",
                        "title": song.title or p.stem,
                        "tuning": tuning_name,
                        "tech_ratio": round(tech_ratio * 100, 1),
                        "total_notes": total_notes
                    })
                    break
        except Exception:
            continue

print(f"\n合計 {len(found_files)} 件のラスボス（Boss Test）ファイルを抽出・隔離しました。")
import json
with open("datasets/solotab26k/boss_test_summary.json", "w", encoding="utf-8") as f:
    json.dump(found_files, f, ensure_ascii=False, indent=2)

for i, item in enumerate(found_files[:15]):
    artist_safe = item['artist'].encode('ascii', 'replace').decode('ascii')
    title_safe = item['title'].encode('ascii', 'replace').decode('ascii')
    print(f"[{i+1}] {artist_safe} - {title_safe} ({item['tuning']}, 奏法率: {item['tech_ratio']}%, 音数: {item['total_notes']})")
