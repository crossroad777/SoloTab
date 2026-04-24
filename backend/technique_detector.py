"""
technique_detector.py — ギターテクニック検出器（拡張版）
================================================================
連続するノートのピッチ・タイミング・弦/フレットパターンから
ギターテクニックを推定する。

検出テクニック (9種):
  - h: ハンマリングオン (同弦、上昇、短間隔、近フレット)
  - p: プリングオフ (同弦、下降、短間隔、近フレット)
  - /: スライドアップ (同弦、上昇、中間隔、2半音以上)
  - \\: スライドダウン (同弦、下降、中間隔、2半音以上)
  - b: ベンド (非常に短い間隔、1-2半音上昇)
  - ~: ビブラート (同弦で微小ピッチ変動が連続)
  - let_ring: レットリング (長い音価 + 次のノートとオーバーラップ)
  - x: ゴーストノート (極低velocity + 短duration)
  - tr: トリル (h/pの高速交互繰り返し)

改善点 (Phase 8):
  - 2パス方式: Pass 1で連続ノート分析、Pass 2で全体パターン分析
  - ビブラート検出: 同弦で微小ピッチ揺れ (±1半音) が3回以上連続
  - レットリング: 長い音価 + 異弦の次のノートと時間的オーバーラップ
  - ゴーストノート: 極低velocity(< 0.1) + 短duration(< 80ms)
  - トリル: 同弦h/pが3回以上高速交互
"""

from typing import List, Dict, Optional


def detect_techniques(notes: List[Dict], bpm: float = 120.0) -> List[Dict]:
    """
    ノートリストにテクニック情報を付与する。
    
    Parameters
    ----------
    notes : list[dict]
        弦/フレット割り当て済みのノートリスト。
    bpm : float
        BPM (テンポに応じた閾値調整に使用)
    
    Returns
    -------
    list[dict]
        各ノートに "technique" キーが追加されたリスト。
    """
    if not notes or len(notes) < 2:
        return notes
    
    # BPM連動閾値
    beat_dur = 60.0 / max(bpm, 40)
    sixteenth = beat_dur / 4
    
    # 閾値をBPMに連動
    HP_MAX_INTERVAL = min(sixteenth * 1.5, 0.15)
    HP_MIN_INTERVAL = 0.01
    SLIDE_MAX_INTERVAL = min(beat_dur * 0.5, 0.30)
    SLIDE_MIN_INTERVAL = 0.02
    BEND_MAX_INTERVAL = min(sixteenth * 0.8, 0.08)
    
    HP_MAX_FRET_DIST = 4
    HP_MAX_SEMITONES = 5
    SLIDE_MIN_SEMITONES = 2
    SLIDE_MAX_SEMITONES = 12
    BEND_SEMITONES = (1, 2)
    MIN_VELOCITY_FOR_HP = 0.15
    
    # ゴーストノート閾値
    GHOST_MAX_VELOCITY = 0.10
    GHOST_MAX_DURATION = 0.08
    
    # レットリング閾値
    LET_RING_MIN_DURATION = beat_dur * 0.8  # 0.8拍以上（アルペジオのベース音に対応）
    
    # 時間順にソート
    sorted_notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))
    
    # 各ノートのテクニックを初期化
    for n in sorted_notes:
        n["technique"] = None
    
    # ===== Pass 0: ゴーストノート検出 (独立判定) =====
    for n in sorted_notes:
        vel = n.get("velocity", 0.5)
        dur = n.get("end", n["start"]) - n["start"]
        if vel < GHOST_MAX_VELOCITY and dur < GHOST_MAX_DURATION:
            n["technique"] = "x"
    
    # ===== Pass 1: 連続2ノート間のパターン分析 =====
    for i in range(len(sorted_notes) - 1):
        curr = sorted_notes[i]
        next_note = sorted_notes[i + 1]
        
        # ゴーストノートは他テクニックの起点/終点にしない
        if curr.get("technique") == "x" or next_note.get("technique") == "x":
            continue
        
        # 基本情報
        interval = next_note["start"] - curr["start"]
        same_string = (curr.get("string", 0) == next_note.get("string", 0) 
                       and curr.get("string", 0) > 0)
        pitch_diff = next_note["pitch"] - curr["pitch"]
        fret_diff = abs(next_note.get("fret", 0) - curr.get("fret", 0))
        
        if not same_string:
            # --- レットリング (異弦) ---
            curr_dur = curr.get("end", curr["start"]) - curr["start"]
            overlap = curr.get("end", curr["start"]) - next_note["start"]
            if (curr_dur >= LET_RING_MIN_DURATION and
                overlap > 0.05 and
                curr.get("technique") is None):
                curr["technique"] = "let_ring"
            continue
        
        # velocityチェック
        next_vel = next_note.get("velocity", 0.5)
        
        # --- ベンド (最優先) ---
        if (pitch_diff in BEND_SEMITONES and
            interval < BEND_MAX_INTERVAL and interval > 0 and
            fret_diff == 0):
            next_note["technique"] = "b"
            continue
        
        # --- ハンマリングオン ---
        if (pitch_diff > 0 and pitch_diff <= HP_MAX_SEMITONES and
            interval < HP_MAX_INTERVAL and interval > HP_MIN_INTERVAL and
            fret_diff <= HP_MAX_FRET_DIST and
            next_vel >= MIN_VELOCITY_FOR_HP):
            next_note["technique"] = "h"
            continue
        
        # --- プリングオフ ---
        if (pitch_diff < 0 and pitch_diff >= -HP_MAX_SEMITONES and
            interval < HP_MAX_INTERVAL and interval > HP_MIN_INTERVAL and
            fret_diff <= HP_MAX_FRET_DIST and
            next_vel >= MIN_VELOCITY_FOR_HP):
            next_note["technique"] = "p"
            continue
        
        # --- スライドアップ ---
        if (pitch_diff >= SLIDE_MIN_SEMITONES and 
            pitch_diff <= SLIDE_MAX_SEMITONES and
            interval < SLIDE_MAX_INTERVAL and interval > SLIDE_MIN_INTERVAL and
            fret_diff >= SLIDE_MIN_SEMITONES):
            next_note["technique"] = "/"
            continue
        
        # --- スライドダウン ---
        if (pitch_diff <= -SLIDE_MIN_SEMITONES and 
            pitch_diff >= -SLIDE_MAX_SEMITONES and
            interval < SLIDE_MAX_INTERVAL and interval > SLIDE_MIN_INTERVAL and
            fret_diff >= SLIDE_MIN_SEMITONES):
            next_note["technique"] = "\\"
            continue
    
    # ===== Pass 2: パターン分析 (トリル・ビブラート) =====
    
    # --- トリル検出: 同弦h/pが3+回高速交互 ---
    i = 0
    while i < len(sorted_notes) - 2:
        n = sorted_notes[i]
        if n.get("technique") in ("h", "p"):
            # 連続するh/p列を検索
            chain = [i]
            j = i + 1
            while (j < len(sorted_notes) and
                   sorted_notes[j].get("technique") in ("h", "p") and
                   sorted_notes[j].get("string") == n.get("string")):
                chain.append(j)
                j += 1
            
            if len(chain) >= 3:
                # ピッチが2値間で交互していればトリル
                pitches = [sorted_notes[k]["pitch"] for k in chain]
                unique_pitches = set(pitches)
                if len(unique_pitches) <= 2:
                    for k in chain:
                        sorted_notes[k]["technique"] = "tr"
            
            i = j
        else:
            i += 1
    
    # --- ビブラート検出: 同弦で微小ピッチ変動が連続 ---
    # すでにテクニック割り当て済みのノートはスキップ
    i = 0
    while i < len(sorted_notes) - 2:
        n = sorted_notes[i]
        if n.get("technique") is not None:
            i += 1
            continue
        
        # 同弦・短間隔・微小ピッチ差 (±1半音) の連続を検索
        chain = [i]
        j = i + 1
        while j < len(sorted_notes):
            nj = sorted_notes[j]
            interval = nj["start"] - sorted_notes[chain[-1]]["start"]
            same_str = (nj.get("string", 0) == n.get("string", 0) and
                       n.get("string", 0) > 0)
            pdiff = abs(nj["pitch"] - n["pitch"])
            
            if same_str and interval < 0.2 and pdiff <= 1 and nj.get("technique") is None:
                chain.append(j)
                j += 1
            else:
                break
        
        if len(chain) >= 3:
            # 3+ノートがピッチ揺れパターン → ビブラート
            for k in chain:
                sorted_notes[k]["technique"] = "~"
        
        i = max(i + 1, j)
    
    # ===== Pass 3: アルペジオベース音のlet_ring =====
    # 低音域（弦4-6）で次のノートと時間的にオーバーラップしているベース音にlet_ringを付与
    ARPEGGIO_BASS_MIN_STRING = 4  # 弦4以下（低音弦）
    ARPEGGIO_MIN_OVERLAP = 0.1   # 次のノートとの最小オーバーラップ（秒）
    for i in range(len(sorted_notes) - 1):
        n = sorted_notes[i]
        if n.get("technique") is not None:
            continue
        
        s = n.get("string", 0)
        if s < ARPEGGIO_BASS_MIN_STRING:
            continue
        
        dur = n.get("end", n["start"]) - n["start"]
        if dur < beat_dur * 0.4:  # 最低0.4拍の持続
            continue
        
        # 次のノートとのオーバーラップをチェック
        next_n = sorted_notes[i + 1]
        overlap = n.get("end", n["start"]) - next_n["start"]
        if overlap >= ARPEGGIO_MIN_OVERLAP and next_n.get("string", 0) != s:
            n["technique"] = "let_ring"
    
    # 統計
    technique_counts = {}
    for n in sorted_notes:
        t = n.get("technique")
        if t:
            technique_counts[t] = technique_counts.get(t, 0) + 1
    
    if technique_counts:
        print(f"[technique_detector] Detected: {technique_counts}")
    
    return sorted_notes


def add_techniques_to_musicxml_notes(notes: List[Dict]) -> List[Dict]:
    """
    テクニック情報をMusicXML用のフォーマットに変換。
    """
    TECHNIQUE_MAP = {
        "h":  {"type": "hammer-on", "text": "H"},
        "p":  {"type": "pull-off", "text": "P"},
        "/":  {"type": "slide", "text": "S", "direction": "up"},
        "\\": {"type": "slide", "text": "S", "direction": "down"},
        "b":  {"type": "bend", "text": "b"},
        "~":  {"type": "vibrato", "text": "~"},
        "let_ring": {"type": "let-ring", "text": "let ring"},
        "x":  {"type": "ghost-note", "text": "x"},
        "tr": {"type": "trill", "text": "tr"},
    }
    
    for n in notes:
        tech = n.get("technique")
        if tech and tech in TECHNIQUE_MAP:
            n["technical_notation"] = TECHNIQUE_MAP[tech]
    
    return notes


if __name__ == "__main__":
    # テスト
    test_notes = [
        {"start": 0.0, "end": 0.5, "pitch": 60, "string": 3, "fret": 5, "velocity": 0.6},
        {"start": 0.08, "end": 0.5, "pitch": 62, "string": 3, "fret": 7, "velocity": 0.5},  # h
        {"start": 0.5, "end": 1.0, "pitch": 65, "string": 2, "fret": 5, "velocity": 0.6},
        {"start": 0.58, "end": 1.0, "pitch": 63, "string": 2, "fret": 3, "velocity": 0.5},  # p
        {"start": 1.0, "end": 1.5, "pitch": 60, "string": 3, "fret": 5, "velocity": 0.6},
        {"start": 1.15, "end": 1.5, "pitch": 64, "string": 3, "fret": 9, "velocity": 0.5},  # /
        {"start": 2.0, "end": 2.03, "pitch": 55, "string": 4, "fret": 5, "velocity": 0.05},  # x (ghost)
        {"start": 3.0, "end": 5.0, "pitch": 60, "string": 3, "fret": 5, "velocity": 0.6},   # let_ring candidate
        {"start": 3.1, "end": 4.0, "pitch": 64, "string": 2, "fret": 5, "velocity": 0.5},
    ]
    
    result = detect_techniques(test_notes, bpm=120)
    for n in result:
        tech = n.get("technique", "-")
        print(f"  t={n['start']:.2f} s{n.get('string', '?')} f{n.get('fret', '?')} "
              f"MIDI={n['pitch']} tech={tech}")

