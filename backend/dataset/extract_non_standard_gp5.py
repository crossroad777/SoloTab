"""
extract_non_standard_gp5.py — 非標準チューニング ＆ 特殊奏法 GP5 抽出・統計スクリプト
================================================================================
26,092件のGP5コレクションから以下を抽出・隔離する:
1. 非標準チューニング (DADGAD, Drop D, Open系, Half Down等)
2. 特殊奏法を含むもの (Tapping, Harmonics, Dead Notes/Percussive)
"""

import os
import sys
import shutil
import pathlib
import json
import guitarpro

# Standard Tuning (MIDI: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64)
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)

TUNING_MAP = {
    (38, 45, 50, 55, 57, 62): "DADGAD",
    (38, 45, 50, 55, 59, 64): "Drop D",
    (36, 43, 48, 53, 57, 62): "Drop C",
    (38, 45, 50, 54, 57, 62): "Open D",
    (38, 43, 50, 55, 59, 62): "Open G",
    (36, 43, 48, 55, 60, 64): "Open C",
    (40, 47, 52, 56, 59, 64): "Open E",
    (38, 45, 50, 55, 59, 62): "Double Drop D",
    (39, 44, 49, 54, 58, 63): "Half Down (Eb)",
    (38, 43, 48, 53, 57, 62): "Full Down (D)",
}

def scan_and_extract():
    search_roots = [
        pathlib.Path("gprotab_downloads"),
        pathlib.Path("../datasets"),
        pathlib.Path("gp_training_data"),
        pathlib.Path("SynthTab-main"),
        pathlib.Path("datasets"),
    ]
    
    dest_dir = pathlib.Path("datasets/non_standard")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    total_scanned = 0
    non_standard_files = []
    tuning_distribution = {}
    technique_counts = {
        "tapping": 0,
        "harmonics": 0,
        "dead_notes": 0,
        "slapping": 0,
        "bends": 0,
        "slides": 0,
    }
    
    seen_files = set()
    
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.gp*"):
            if p.suffix.lower() not in (".gp3", ".gp4", ".gp5"):
                continue
            if p.name in seen_files:
                continue
            seen_files.add(p.name)
            total_scanned += 1
            
            try:
                song = guitarpro.parse(str(p))
                is_non_standard = False
                has_special_tech = False
                detected_tuning = "Standard"
                
                file_techs = set()
                
                for t in song.tracks:
                    if len(t.strings) != 6 or t.isPercussionTrack:
                        continue
                        
                    tun = tuple(s.value for s in t.strings)
                    if tun != STANDARD_TUNING:
                        is_non_standard = True
                        detected_tuning = TUNING_MAP.get(tun, "Other Alternate")
                        
                    for m in t.measures:
                        for v in m.voices:
                            for b in v.beats:
                                # Tapping
                                if hasattr(b.effect, 'tapping') and b.effect.tapping:
                                    file_techs.add("tapping")
                                    has_special_tech = True
                                # Slapping / Popping
                                if hasattr(b.effect, 'slapEffect') and b.effect.slapEffect and str(b.effect.slapEffect) != 'SlapEffect.none':
                                    file_techs.add("slapping")
                                    has_special_tech = True
                                
                                for n in b.notes:
                                    # Dead Note
                                    if n.type == guitarpro.NoteType.dead:
                                        file_techs.add("dead_notes")
                                        has_special_tech = True
                                    # Harmonics
                                    if hasattr(n.effect, 'harmonic') and n.effect.harmonic:
                                        file_techs.add("harmonics")
                                        has_special_tech = True
                                    # Bend
                                    if hasattr(n.effect, 'bend') and n.effect.bend:
                                        file_techs.add("bends")
                                    # Slide
                                    if hasattr(n.effect, 'slides') and n.effect.slides:
                                        file_techs.add("slides")
                                        
                if is_non_standard or has_special_tech:
                    shutil.copy2(str(p), str(dest_dir / p.name))
                    for tech in file_techs:
                        technique_counts[tech] += 1
                    tuning_distribution[detected_tuning] = tuning_distribution.get(detected_tuning, 0) + 1
                    
                    non_standard_files.append({
                        "filename": p.name,
                        "tuning": detected_tuning,
                        "techniques": list(file_techs),
                        "artist": song.artist or "Unknown",
                        "title": song.title or p.stem,
                    })
            except Exception:
                pass

    summary = {
        "total_scanned": total_scanned,
        "total_extracted": len(non_standard_files),
        "tuning_distribution": tuning_distribution,
        "technique_counts": technique_counts,
        "files": non_standard_files[:50]
    }
    
    with open("datasets/non_standard_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print(f"=== 非標準チューニング ＆ 特殊奏法 抽出結果 ===")
    print(f"総スキャン: {total_scanned} 件")
    print(f"抽出ファイル: {len(non_standard_files)} 件 (datasets/non_standard/ に保存)")
    print("\nチューニング分布:")
    for tun, cnt in sorted(tuning_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tun:<18s}: {cnt:>5d} 件")
    print("\n特殊奏法検出件数:")
    for tech, cnt in sorted(technique_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tech:<18s}: {cnt:>5d} 件")

if __name__ == "__main__":
    scan_and_extract()
