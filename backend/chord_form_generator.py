"""
chord_form_generator.py — ギター典型フォームDB生成
===================================================
CAGEDシステム + 標準コードフォームを基盤に、
12音 × コード種 × ポジションの網羅的な押弦パターンDBを生成する。

出力形式: JSON
各エントリ = {
    "chord": "Am",           # コード名
    "root": 9,               # ルート音のピッチクラス (0=C, 1=C#, ..., 11=B)
    "quality": "minor",      # コード種
    "frets": [0, 0, 2, 2, 1, 0],  # 6弦→1弦のフレット番号 (-1=ミュート, 0=開放)
    "position": 0,           # ポジション（最低フレット）
    "bass_string": 5,        # ベース弦 (6=6弦, 5=5弦, ...)
    "notes": [40, 45, 50, ...],  # 各弦のMIDIノート (-1=ミュート)
}

坂井論文(2024)の典型フォーム3,658個に相当する規模を目指す。
"""

import json
from typing import List, Dict, Tuple, Optional

# 標準チューニング (6弦→1弦): E2, A2, D3, G3, B3, E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

# 音名マッピング
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# コード構成音（ルートからの半音数）
CHORD_INTERVALS = {
    "major":     [0, 4, 7],
    "minor":     [0, 3, 7],
    "7":         [0, 4, 7, 10],
    "m7":        [0, 3, 7, 10],
    "maj7":      [0, 4, 7, 11],
    "dim":       [0, 3, 6],
    "dim7":      [0, 3, 6, 9],
    "aug":       [0, 4, 8],
    "sus4":      [0, 5, 7],
    "sus2":      [0, 2, 7],
    "add9":      [0, 4, 7, 14],
    "m7b5":      [0, 3, 6, 10],
    "6":         [0, 4, 7, 9],
    "m6":        [0, 3, 7, 9],
    "9":         [0, 4, 7, 10, 14],
    "m9":        [0, 3, 7, 10, 14],
}

# =============================================================================
# 基本フォーム定義（開放弦コード + バレーテンプレート）
# =============================================================================
# 形式: (frets, muted_strings, base_root_pc, quality)
# frets: [6弦, 5弦, 4弦, 3弦, 2弦, 1弦]
# muted_strings: ミュートする弦のインデックス (0-based, 0=6弦)

# --- 開放弦コード ---
OPEN_CHORDS = [
    # Major
    {"frets": [-1, 3, 2, 0, 1, 0], "root_pc": 0,  "quality": "major",  "name": "C"},
    {"frets": [-1, -1, 0, 2, 3, 2], "root_pc": 2,  "quality": "major",  "name": "D"},
    {"frets": [0, 2, 2, 1, 0, 0], "root_pc": 4,  "quality": "major",  "name": "E"},
    {"frets": [1, 1, 2, 3, 3, 1], "root_pc": 5,  "quality": "major",  "name": "F"},
    {"frets": [3, 2, 0, 0, 0, 3], "root_pc": 7,  "quality": "major",  "name": "G"},
    {"frets": [-1, 0, 2, 2, 2, 0], "root_pc": 9,  "quality": "major",  "name": "A"},
    # Minor
    {"frets": [-1, 3, 2, 0, 1, 0], "root_pc": 0,  "quality": "minor",  "name": "Cm_partial"},
    {"frets": [-1, -1, 0, 2, 3, 1], "root_pc": 2,  "quality": "minor",  "name": "Dm"},
    {"frets": [0, 2, 2, 0, 0, 0], "root_pc": 4,  "quality": "minor",  "name": "Em"},
    {"frets": [-1, 0, 2, 2, 1, 0], "root_pc": 9,  "quality": "minor",  "name": "Am"},
    # 7th
    {"frets": [-1, 3, 2, 3, 1, 0], "root_pc": 0,  "quality": "7",  "name": "C7"},
    {"frets": [-1, -1, 0, 2, 1, 2], "root_pc": 2,  "quality": "7",  "name": "D7"},
    {"frets": [0, 2, 0, 1, 0, 0], "root_pc": 4,  "quality": "7",  "name": "E7"},
    {"frets": [3, 2, 0, 0, 0, 1], "root_pc": 7,  "quality": "7",  "name": "G7"},
    {"frets": [-1, 0, 2, 0, 2, 0], "root_pc": 9,  "quality": "7",  "name": "A7"},
    {"frets": [-1, 0, 2, 0, 1, 0], "root_pc": 9,  "quality": "m7",  "name": "Am7"},
    {"frets": [0, 2, 0, 0, 0, 0], "root_pc": 4,  "quality": "m7",  "name": "Em7"},
    # maj7
    {"frets": [-1, 3, 2, 0, 0, 0], "root_pc": 0,  "quality": "maj7",  "name": "Cmaj7"},
    {"frets": [-1, -1, 0, 2, 2, 2], "root_pc": 2,  "quality": "maj7",  "name": "Dmaj7"},
    {"frets": [-1, 0, 2, 1, 2, 0], "root_pc": 9,  "quality": "maj7",  "name": "Amaj7"},
    # sus
    {"frets": [-1, -1, 0, 2, 3, 3], "root_pc": 2,  "quality": "sus4",  "name": "Dsus4"},
    {"frets": [0, 2, 2, 2, 0, 0], "root_pc": 4,  "quality": "sus4",  "name": "Esus4"},
    {"frets": [-1, 0, 2, 2, 3, 0], "root_pc": 9,  "quality": "sus4",  "name": "Asus4"},
    {"frets": [-1, 0, 2, 2, 0, 0], "root_pc": 9,  "quality": "sus2",  "name": "Asus2"},
    # dim / aug
    {"frets": [-1, -1, 0, 1, 0, 1], "root_pc": 2,  "quality": "dim",  "name": "Ddim"},
    {"frets": [-1, 0, 1, 2, 2, -1], "root_pc": 9,  "quality": "dim",  "name": "Adim"},
    # 6th
    {"frets": [-1, 0, 2, 2, 2, 2], "root_pc": 9,  "quality": "6",  "name": "A6"},
    # add9
    {"frets": [-1, 3, 2, 0, 3, 0], "root_pc": 0,  "quality": "add9",  "name": "Cadd9"},
]

# --- バレーコード テンプレート（移調可能な形） ---
# base_fret=0 として定義し、移調時に全フレットにオフセットを加算
BARRE_TEMPLATES = [
    # E-shape major (ルート=6弦)
    {"shape": [0, 2, 2, 1, 0, 0], "root_string": 0, "root_offset": 0, "quality": "major", "name": "E-shape"},
    # E-shape minor
    {"shape": [0, 2, 2, 0, 0, 0], "root_string": 0, "root_offset": 0, "quality": "minor", "name": "Em-shape"},
    # E-shape 7
    {"shape": [0, 2, 0, 1, 0, 0], "root_string": 0, "root_offset": 0, "quality": "7", "name": "E7-shape"},
    # E-shape m7
    {"shape": [0, 2, 0, 0, 0, 0], "root_string": 0, "root_offset": 0, "quality": "m7", "name": "Em7-shape"},
    # E-shape maj7
    {"shape": [0, 2, 1, 1, 0, 0], "root_string": 0, "root_offset": 0, "quality": "maj7", "name": "Emaj7-shape"},
    # A-shape major (ルート=5弦)
    {"shape": [-1, 0, 2, 2, 2, 0], "root_string": 1, "root_offset": 0, "quality": "major", "name": "A-shape"},
    # A-shape minor
    {"shape": [-1, 0, 2, 2, 1, 0], "root_string": 1, "root_offset": 0, "quality": "minor", "name": "Am-shape"},
    # A-shape 7
    {"shape": [-1, 0, 2, 0, 2, 0], "root_string": 1, "root_offset": 0, "quality": "7", "name": "A7-shape"},
    # A-shape m7
    {"shape": [-1, 0, 2, 0, 1, 0], "root_string": 1, "root_offset": 0, "quality": "m7", "name": "Am7-shape"},
    # C-shape major (ルート=5弦)
    {"shape": [-1, 3, 2, 0, 1, 0], "root_string": 1, "root_offset": 3, "quality": "major", "name": "C-shape"},
    # D-shape major (ルート=4弦)
    {"shape": [-1, -1, 0, 2, 3, 2], "root_string": 2, "root_offset": 0, "quality": "major", "name": "D-shape"},
    # D-shape minor
    {"shape": [-1, -1, 0, 2, 3, 1], "root_string": 2, "root_offset": 0, "quality": "minor", "name": "Dm-shape"},
    # Power chord (5th)
    {"shape": [0, 2, 2, -1, -1, -1], "root_string": 0, "root_offset": 0, "quality": "5", "name": "power5-E"},
    {"shape": [-1, 0, 2, 2, -1, -1], "root_string": 1, "root_offset": 0, "quality": "5", "name": "power5-A"},
]

# コード種の表示名マッピング
QUALITY_SUFFIX = {
    "major": "", "minor": "m", "7": "7", "m7": "m7", "maj7": "maj7",
    "dim": "dim", "dim7": "dim7", "aug": "aug", "sus4": "sus4", "sus2": "sus2",
    "add9": "add9", "m7b5": "m7b5", "6": "6", "m6": "m6", "9": "9", "m9": "m9",
    "5": "5",
}


def frets_to_midi(frets: List[int], tuning: List[int] = STANDARD_TUNING) -> List[int]:
    """フレット配列をMIDIノートに変換。-1はミュート。"""
    notes = []
    for i, f in enumerate(frets):
        if f < 0:
            notes.append(-1)
        else:
            notes.append(tuning[i] + f)
    return notes


def get_sounding_notes_pc(frets: List[int], tuning: List[int] = STANDARD_TUNING) -> List[int]:
    """発音する弦のピッチクラスを返す。"""
    pcs = []
    for i, f in enumerate(frets):
        if f >= 0:
            pcs.append((tuning[i] + f) % 12)
    return pcs


def validate_form(frets: List[int], max_span: int = 4, max_fret: int = 14) -> bool:
    """押弦パターンが物理的に演奏可能かチェック。"""
    pressed = [f for f in frets if f > 0]
    if not pressed:
        return True  # 全開放 or 全ミュートは有効
    if max(pressed) > max_fret:
        return False
    span = max(pressed) - min(pressed)
    return span <= max_span


def transpose_barre(template: dict, semitones: int) -> Optional[dict]:
    """バレーテンプレートを指定半音数だけ移調する。"""
    shape = template["shape"]
    new_frets = []
    for f in shape:
        if f < 0:
            new_frets.append(-1)
        else:
            new_frets.append(f + semitones)
    
    if not validate_form(new_frets):
        return None
    
    # ルートのピッチクラスを計算
    root_string = template["root_string"]
    root_fret = new_frets[root_string]
    if root_fret < 0:
        return None
    root_pc = (STANDARD_TUNING[root_string] + root_fret) % 12
    
    chord_name = NOTE_NAMES[root_pc] + QUALITY_SUFFIX.get(template["quality"], "")
    
    return {
        "chord": chord_name,
        "root_pc": root_pc,
        "quality": template["quality"],
        "frets": new_frets,
        "position": min(f for f in new_frets if f > 0) if any(f > 0 for f in new_frets) else 0,
        "source": template["name"],
    }


def generate_partial_voicings(form: dict, max_removals: int = 2) -> List[dict]:
    """
    完全なフォームから弦を1-2本ミュートした部分ヴォイシングを生成。
    坂井論文3.2.2の「押弦箇所を減らす処理」に相当。
    """
    partials = []
    frets = form["frets"]
    sounding = [i for i, f in enumerate(frets) if f >= 0]
    
    if len(sounding) <= 3:
        return []  # 3弦以下なら部分化不要
    
    # 1弦ミュート
    for i in sounding:
        new_frets = frets.copy()
        new_frets[i] = -1
        remaining = [f for f in new_frets if f >= 0]
        if len(remaining) >= 2 and validate_form(new_frets):
            pcs = get_sounding_notes_pc(new_frets)
            # ルート音が残っているか確認
            if form["root_pc"] in pcs:
                partials.append({
                    "chord": form["chord"],
                    "root_pc": form["root_pc"],
                    "quality": form["quality"],
                    "frets": new_frets,
                    "position": form.get("position", 0),
                    "source": form.get("source", "partial"),
                    "partial": True,
                })
    
    return partials


def generate_database() -> List[dict]:
    """典型フォームDBを生成する。"""
    forms = []
    seen = set()  # 重複排除用
    
    def add_form(f: dict):
        key = (f["chord"], tuple(f["frets"]))
        if key not in seen:
            seen.add(key)
            # MIDI ノートを計算
            f["notes"] = frets_to_midi(f["frets"])
            # ベース弦を特定
            for i in range(5, -1, -1):  # 6弦から探す
                if f["frets"][i] >= 0:
                    f["bass_string"] = 6 - i  # 6=6弦, 1=1弦
                    break
            else:
                f["bass_string"] = 0
            forms.append(f)
    
    # --- 1. 開放弦コード ---
    for oc in OPEN_CHORDS:
        root_name = NOTE_NAMES[oc["root_pc"]]
        chord_name = root_name + QUALITY_SUFFIX.get(oc["quality"], "")
        form = {
            "chord": chord_name,
            "root_pc": oc["root_pc"],
            "quality": oc["quality"],
            "frets": oc["frets"],
            "position": 0,
            "source": "open",
        }
        add_form(form)
    
    # --- 2. バレーコード（各テンプレートを12音に移調） ---
    for tmpl in BARRE_TEMPLATES:
        for semitones in range(0, 13):
            form = transpose_barre(tmpl, semitones)
            if form is not None:
                add_form(form)
    
    # --- 3. 部分ヴォイシング ---
    base_forms = forms.copy()
    for form in base_forms:
        partials = generate_partial_voicings(form)
        for p in partials:
            add_form(p)
    
    # --- 4. 追加の一般的なヴォイシング（手動定義） ---
    # ハイポジションの一般的なコードフォーム
    extra_forms = [
        # Fmaj7 (1st pos)
        {"frets": [-1, -1, 3, 2, 1, 0], "root_pc": 5, "quality": "maj7"},
        # Bm
        {"frets": [-1, 2, 4, 4, 3, 2], "root_pc": 11, "quality": "minor"},
        # B7
        {"frets": [-1, 2, 1, 2, 0, 2], "root_pc": 11, "quality": "7"},
        # F#m
        {"frets": [2, 4, 4, 2, 2, 2], "root_pc": 6, "quality": "minor"},
        # C#m
        {"frets": [-1, 4, 6, 6, 5, 4], "root_pc": 1, "quality": "minor"},
        # G#m / Abm
        {"frets": [4, 6, 6, 4, 4, 4], "root_pc": 8, "quality": "minor"},
        # Bb
        {"frets": [-1, 1, 3, 3, 3, 1], "root_pc": 10, "quality": "major"},
        # Eb
        {"frets": [-1, -1, 1, 3, 4, 3], "root_pc": 3, "quality": "major"},
        # Cm (barre)
        {"frets": [-1, 3, 5, 5, 4, 3], "root_pc": 0, "quality": "minor"},
        # Fm
        {"frets": [1, 3, 3, 1, 1, 1], "root_pc": 5, "quality": "minor"},
        # Gm
        {"frets": [3, 5, 5, 3, 3, 3], "root_pc": 7, "quality": "minor"},
        # Dm7
        {"frets": [-1, -1, 0, 2, 1, 1], "root_pc": 2, "quality": "m7"},
        # G6
        {"frets": [3, 2, 0, 0, 0, 0], "root_pc": 7, "quality": "6"},
        # Cadd9
        {"frets": [-1, 3, 2, 0, 3, 0], "root_pc": 0, "quality": "add9"},
        # Em9
        {"frets": [0, 2, 0, 0, 0, 2], "root_pc": 4, "quality": "m9"},
        # Dsus2
        {"frets": [-1, -1, 0, 2, 3, 0], "root_pc": 2, "quality": "sus2"},
    ]
    
    for ef in extra_forms:
        root_name = NOTE_NAMES[ef["root_pc"]]
        chord_name = root_name + QUALITY_SUFFIX.get(ef["quality"], "")
        form = {
            "chord": chord_name,
            "root_pc": ef["root_pc"],
            "quality": ef["quality"],
            "frets": ef["frets"],
            "position": min(f for f in ef["frets"] if f > 0) if any(f > 0 for f in ef["frets"]) else 0,
            "source": "extra",
        }
        add_form(form)
    
    return forms


def build_lookup(forms: List[dict]) -> dict:
    """コード名→フォームリストのルックアップテーブルを構築。"""
    lookup = {}
    for form in forms:
        chord = form["chord"]
        if chord not in lookup:
            lookup[chord] = []
        lookup[chord].append(form)
    return lookup


if __name__ == "__main__":
    print("Generating guitar chord form database...")
    forms = generate_database()
    
    # 統計
    qualities = {}
    for f in forms:
        q = f["quality"]
        qualities[q] = qualities.get(q, 0) + 1
    
    print(f"\nTotal forms: {len(forms)}")
    print(f"\nBy quality:")
    for q, count in sorted(qualities.items(), key=lambda x: -x[1]):
        print(f"  {q}: {count}")
    
    # コード数
    chords = set(f["chord"] for f in forms)
    print(f"\nUnique chords: {len(chords)}")
    
    # JSON保存
    output_path = "chord_forms_db.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(forms, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # ルックアップ保存
    lookup = build_lookup(forms)
    lookup_path = "chord_forms_lookup.json"
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    
    print(f"Lookup saved to {lookup_path}")
    
    # サンプル表示
    print("\n--- Sample forms ---")
    for chord_name in ["C", "Am", "G", "Em", "D", "F", "B7", "F#m"]:
        if chord_name in lookup:
            print(f"\n{chord_name} ({len(lookup[chord_name])} voicings):")
            for v in lookup[chord_name][:3]:
                frets_str = "".join(str(f) if f >= 0 else "x" for f in v["frets"])
                print(f"  [{frets_str}] pos={v['position']} src={v['source']}")
