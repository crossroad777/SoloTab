from __future__ import annotations
"""
note_filter.py — ノート密度フィルタリング
========================================
TABの実用性を高めるため、検出ノートを適切に間引く。
各フィルタは独立して適用可能。
"""

import logging
from bisect import bisect_right
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def filter_by_velocity(notes: List[Dict], threshold: float = 0.3) -> List[Dict]:
    """
    velocity閾値フィルタ。
    閾値未満の弱い音を除去する。
    
    Args:
        notes: ノートリスト
        threshold: velocity閾値 (0.0-1.0)。デフォルト0.3 (≈38/127)
    """
    filtered = [n for n in notes if n.get("velocity", 1.0) >= threshold]
    removed = len(notes) - len(filtered)
    if removed:
        logger.info(f"[velocity_filter] Removed {removed} notes (v < {threshold})")
    return filtered


def filter_by_min_duration(notes: List[Dict], min_duration_sec: float = 0.05) -> List[Dict]:
    """
    最小音価フィルタ。
    極端に短い音（50ms以下）を除去する。
    ゴーストノートや誤検出を排除。
    
    Args:
        notes: ノートリスト
        min_duration_sec: 最小持続時間（秒）。デフォルト0.05s (50ms)
    """
    filtered = []
    for n in notes:
        duration = n.get("end", n.get("start", 0)) - n.get("start", 0)
        if duration >= min_duration_sec:
            filtered.append(n)
    removed = len(notes) - len(filtered)
    if removed:
        logger.info(f"[min_duration_filter] Removed {removed} notes (< {min_duration_sec*1000:.0f}ms)")
    return filtered


def filter_harmonics(notes: List[Dict], time_tolerance: float = 0.20,
                     velocity_ratio: float = 0.70,
                     tuning: Optional[List[int]] = None) -> List[Dict]:
    """
    倍音フィルタ v3 — 基音グルーピング + メロディ保護。
    
    v3改善点:
    - 開放弦ピッチ保護: E4/B3/G3等の開放弦と同ピッチの音はメロディの可能性が
      高いため、velocity基準を緩和して保護する
    - 倍音次数依存の除去閾値: 2倍音(オクターブ)は強い共鳴なので厳格、
      4倍音以上は弱いので高velocity=実音として保持
    - メロディ文脈保護: 前後にスケール的音程関係がある音はメロディとして保護
    """
    if not notes:
        return notes
    
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning
    
    # 開放弦ピッチセット（これらのピッチはメロディとしても頻繁に使われる）
    open_string_pitches = set(tuning)
    # オクターブ上の開放弦ピッチも保護（例: E4=64+12=76は少ないが、既にtuningに含む）
    
    # ピッチ→周波数変換
    def midi_to_hz(midi):
        return 440.0 * (2 ** ((midi - 69) / 12))
    
    # 倍音次数ごとの除去閾値
    # v2(F1=0.82)が最善だったため、全倍音を統一閾値で処理
    HARMONIC_VEL_RATIOS = {
        2: 0.70,   # オクターブ
        3: 0.70,   # 5度+オクターブ
        4: 0.70,   # 2オクターブ
        5: 0.70,   # 3度+2オクターブ
        6: 0.70,   # 5度+2オクターブ
    }
    
    NATURAL_HARMONIC_INTERVALS = {12, 19, 24, 28, 31}
    
    notes_sorted = sorted(notes, key=lambda n: (n["start"], n.get("pitch", 0)))
    keep = set(range(len(notes_sorted)))
    harmonic_indices = set()
    
    # メロディ文脈チェック用のヘルパー
    def _has_melody_context(idx, notes_list, keep_set):
        """前後のノートとスケール的音程関係があればメロディの一部"""
        note = notes_list[idx]
        pitch = note.get("pitch", 0)
        t = note["start"]
        
        # 前後1.2秒以内のノートでピッチが近いもの（半音1-5の差）を探す
        neighbors = 0
        pitch_directions = []  # ピッチの上下方向を記録
        for di in range(-7, 8):
            ni = idx + di
            if ni < 0 or ni >= len(notes_list) or ni == idx or ni not in keep_set:
                continue
            nn = notes_list[ni]
            if abs(nn["start"] - t) > 1.2:
                continue
            interval = abs(nn.get("pitch", 0) - pitch)
            if 1 <= interval <= 5:  # 半音〜完全４度
                neighbors += 1
                # ピッチ方向の一貫性チェック用
                direction = nn.get("pitch", 0) - pitch
                if nn["start"] < t:
                    pitch_directions.append(-direction)  # 前の音からの動き
                else:
                    pitch_directions.append(direction)   # 次の音への動き
        
        # 2つ以上の隣接音があれば強いメロディ文脈
        if neighbors >= 2:
            return True
        # 1つでも、ピッチ方向が一貫していればメロディ
        if neighbors >= 1:
            return True
        return False
    
    # === Phase 0: B4 (MIDI 71) 共鳴フィルタ（メロディ保護付き） ===
    # B4はE2/B3の倍音として誤検出されやすいが、メロディ音としても頻出。
    # メロディ文脈（近傍にステップ関係の音がある）があれば保護する。
    for i in range(len(notes_sorted)):
        if i not in keep:
            continue
        note = notes_sorted[i]
        if note.get("pitch", 0) == 71:
            t_b4 = note["start"]
            vel_b4 = note.get("velocity", 1.0)
            
            # velocity 0.65以上は保護（実際にピッキングされた音）
            if vel_b4 >= 0.65:
                continue
            
            # メロディ文脈チェック：近傍にステップ関係(±1-5半音)の別ピッチがあれば保護
            if _has_melody_context(i, notes_sorted, keep):
                continue

            # 共鳴判定: 近傍にE2/B3等の基音がある+velocityが低い場合のみ除去
            has_resonator = False
            resonator_pitch = None
            for j in range(max(0, i-15), min(len(notes_sorted), i+15)):
                if j == i or j not in keep:
                    continue
                other = notes_sorted[j]
                dt = other["start"] - t_b4
                if -0.10 <= dt <= 0.05:
                    if other.get("pitch", 0) in [40, 47, 52, 59]:
                        has_resonator = True
                        resonator_pitch = other.get("pitch", 0)
                        break
            
            if has_resonator:
                logger.info(f"[note_filter] B4 forced resonance removal from {resonator_pitch} at {t_b4:.2f}s (vel={vel_b4:.2f})")
                keep.discard(i)

    # === Phase 1: 基音グルーピング方式 ===
    i = 0
    while i < len(notes_sorted):
        if i not in keep:
            i += 1
            continue
        
        group_start = notes_sorted[i]["start"]
        group = [i]
        j = i + 1
        while j < len(notes_sorted) and notes_sorted[j]["start"] - group_start <= time_tolerance:
            if j in keep:
                group.append(j)
            j += 1
        
        if len(group) < 2:
            i += 1
            continue
        
        group.sort(key=lambda idx: notes_sorted[idx].get("pitch", 0))
        
        for base_pos in range(len(group)):
            base_idx = group[base_pos]
            if base_idx not in keep:
                continue
            
            base_note = notes_sorted[base_idx]
            base_pitch = base_note.get("pitch", 0)
            base_vel = base_note.get("velocity", 1.0)
            base_hz = midi_to_hz(base_pitch)
            
            for upper_pos in range(base_pos + 1, len(group)):
                upper_idx = group[upper_pos]
                if upper_idx not in keep:
                    continue
                
                upper_note = notes_sorted[upper_idx]
                upper_pitch = upper_note.get("pitch", 0)
                upper_vel = upper_note.get("velocity", 1.0)
                upper_hz = midi_to_hz(upper_pitch)
                
                interval = upper_pitch - base_pitch
                
                # 倍音チェック: 何次倍音か判定
                detected_harmonic_n = 0
                for harmonic_n in [2, 3, 4, 5, 6]:
                    expected_hz = base_hz * harmonic_n
                    if abs(upper_hz - expected_hz) / expected_hz < 0.06:
                        detected_harmonic_n = harmonic_n
                        break
                
                if detected_harmonic_n == 0:
                    continue
                
                # === 保護チェック ===
                
                # 1. 開放弦ピッチ保護: 開放弦と同じピッチは除去しにくくする
                is_open_string = upper_pitch in open_string_pitches
                
                # 2. メロディ文脈保護: 前後にスケール的な隣接音があればメロディ
                has_context = _has_melody_context(upper_idx, notes_sorted, keep)
                
                # 3. 倍音次数依存の閾値
                effective_ratio = HARMONIC_VEL_RATIOS.get(detected_harmonic_n, velocity_ratio)
                
                # 保護ボーナス: 3倍音以上に適用 (2倍音=オクターブは強い共鳴なので厳格に除去)
                if detected_harmonic_n >= 3:
                    if is_open_string:
                        effective_ratio *= 0.7  # 開放弦保護
                    if has_context:
                        effective_ratio *= 0.7  # メロディ保護
                
                # ナチュラルハーモニクスの可能性チェック
                if interval in NATURAL_HARMONIC_INTERVALS:
                    if upper_vel <= base_vel * 1.2:
                        harmonic_indices.add(upper_idx)
                    elif base_vel <= upper_vel * 1.2:
                        harmonic_indices.add(base_idx)
                else:
                    # 倍音共鳴判定
                    if upper_vel <= base_vel * effective_ratio:
                        keep.discard(upper_idx)
                    elif not is_open_string and not has_context:
                        # 保護なし + 境界域: 音価も考慮
                        if upper_vel < base_vel * 0.90:
                            upper_dur = upper_note.get("end", upper_note["start"]) - upper_note["start"]
                            base_dur = base_note.get("end", base_note["start"]) - base_note["start"]
                            if upper_dur < base_dur * 0.7:
                                keep.discard(upper_idx)
        
        i += 1
    
    # === Phase 2: ペアワイズ逆方向チェック ===
    # 基音が後から来るケース（低音の検出遅延）にも対応
    for i in range(len(notes_sorted)):
        if i not in keep:
            continue
        n1 = notes_sorted[i]
        p1 = n1.get("pitch", 0)
        v1 = n1.get("velocity", 1.0)
        f1 = midi_to_hz(p1)
        
        for j in range(i + 1, len(notes_sorted)):
            if j not in keep:
                continue
            n2 = notes_sorted[j]
            if n2["start"] - n1["start"] > time_tolerance:
                break
            
            p2 = n2.get("pitch", 0)
            v2 = n2.get("velocity", 1.0)
            f2 = midi_to_hz(p2)
            
            # p1=検出済みノート, p2=その近傍ノート
            # 基音(低)と倍音(高)の関係をチェック
            lower_idx, upper_idx = (i, j) if p1 < p2 else (j, i)
            lp, up = notes_sorted[lower_idx].get("pitch", 0), notes_sorted[upper_idx].get("pitch", 0)
            lv, uv = notes_sorted[lower_idx].get("velocity", 1.0), notes_sorted[upper_idx].get("velocity", 1.0)
            lf, uf = midi_to_hz(lp), midi_to_hz(up)
            
            # ufがlfの倍音かチェック
            is_harmonic = False
            for harmonic_n in [2, 3, 4, 5, 6]:
                expected = lf * harmonic_n
                if abs(uf - expected) / expected < 0.06:
                    is_harmonic = True
                    break
            
            if is_harmonic:
                interval = up - lp
                if interval not in NATURAL_HARMONIC_INTERVALS:
                    # 倍音(高音側)のみを削除対象とする。基音(低音側)は絶対に残す。
                    # 特に従前、v1(低音) <= v2(高音) の時に低音を消していたバグを修正。
                    if uv <= lv * velocity_ratio:
                        keep.discard(upper_idx)
                    elif lp <= 45: # E2-A2付近の極低音は、高音がある程度強くても倍音とみなして高音側を消す
                        if uv <= lv * 1.2: # 低音の1.2倍程度までの高音はゴーストとみなす
                             keep.discard(upper_idx)
    
    # ハーモニクスマーク
    marked = 0
    for idx in harmonic_indices:
        if idx in keep and notes_sorted[idx].get("technique") is None:
            notes_sorted[idx]["technique"] = "harmonic"
            marked += 1
    
    filtered = [notes_sorted[i] for i in sorted(keep)]
    removed = len(notes) - len(filtered)
    if removed or marked:
        logger.info(f"[harmonics_filter] Removed {removed} resonance notes, "
                    f"marked {marked} as natural harmonics")
    return filtered


def filter_position_window(notes: List[Dict], window_size: int = 7,
                           min_notes_for_position: int = 4) -> List[Dict]:
    """
    ポジション固定ウィンドウ。
    短い時間範囲内でのフレットジャンプを抑制し、
    演奏可能性を高める。
    
    直前のノート群から推定されるポジション（中央フレット）から
    window_sizeフレット以上離れたノートを、より近いポジションに
    リマップする（同一ピッチで別の弦を探す）。
    
    Args:
        notes: ノートリスト（startでソート済み前提）
        window_size: ポジションウィンドウ幅（フレット数）
        min_notes_for_position: ポジション推定に必要な最小ノート数
    """
    if len(notes) < min_notes_for_position:
        return notes
    
    result = list(notes)
    
    # スライディングウィンドウでポジションを推定
    window_notes = 12  # 直近12ノートでポジション推定（ソロギター向けにより長い文脈）
    
    for i, note in enumerate(result):
        if i < min_notes_for_position:
            continue
        
        # 直前のノート群からポジション推定
        recent = result[max(0, i - window_notes):i]
        recent_frets = [n.get("fret", 0) for n in recent if n.get("fret") is not None and n.get("fret", 0) > 0]
        
        if len(recent_frets) < 2:
            continue
        
        center_fret = sum(recent_frets) / len(recent_frets)
        current_fret = note.get("fret", 0)
        
        # ウィンドウ外のノートを検出
        if current_fret > 0 and abs(current_fret - center_fret) > window_size:
            # 同じピッチで、ウィンドウ内に収まる別の弦-フレットを探す
            pitch = note.get("pitch", 0)
            best_alt = None
            best_dist = abs(current_fret - center_fret)
            
            # 各弦で同じピッチが弾けるか確認
            for s in range(1, 7):
                open_pitch = _get_string_open_pitch(s)
                if open_pitch is None:
                    continue
                alt_fret = pitch - open_pitch
                if 0 <= alt_fret <= 19:  # MAX_FRET=19と統一
                    dist = abs(alt_fret - center_fret)
                    if dist < best_dist:
                        best_dist = dist
                        best_alt = (s, alt_fret)
            
            if best_alt:
                note["string"] = best_alt[0]
                note["fret"] = best_alt[1]
    
    return result


def _get_string_open_pitch(string_num: int) -> int:
    """スタンダードチューニングでの各弦のオープンピッチ"""
    # Standard: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64
    pitches = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    return pitches.get(string_num)


def filter_close_duplicates(notes: List[Dict], time_tolerance: float = 0.1) -> List[Dict]:
    """
    近接重複ノート除去。
    同じピッチが100ms以内に複数検出された場合、velocityの高い方を残す。
    """
    if not notes:
        return notes
    
    notes_sorted = sorted(notes, key=lambda n: (n.get("pitch", 0), n["start"]))
    keep = []
    i = 0
    while i < len(notes_sorted):
        best = notes_sorted[i]
        j = i + 1
        while (j < len(notes_sorted) and
               notes_sorted[j].get("pitch") == best.get("pitch") and
               abs(notes_sorted[j]["start"] - best["start"]) < time_tolerance):
            if notes_sorted[j].get("velocity", 0) > best.get("velocity", 0):
                best = notes_sorted[j]
            j += 1
        keep.append(best)
        i = j
    
    removed = len(notes) - len(keep)
    if removed:
        logger.info(f"[close_duplicates] Removed {removed} duplicate notes")
    return sorted(keep, key=lambda n: (n["start"], n.get("pitch", 0)))


def filter_open_string_resonance(notes: List[Dict],
                                  time_tolerance: float = 0.15,
                                  velocity_ratio: float = 0.35) -> List[Dict]:
    """
    開放弦共鳴ノート除去フィルタ（強化版）。
    
    ギターでは弦を弾くと他の開放弦が共鳴する。特に:
    - E2を弾く → E3, G3が共鳴 (開放弦E3, G3)
    - B3を弾く → G3が共鳴
    
    これらの共鳴ノートはモデルが高velocityで検出することがあるため、
    velocity比率ではなく「共起パターン」で判定する。
    """
    if not notes or len(notes) < 2:
        return notes
    
    notes_sorted = sorted(notes, key=lambda n: (n["start"], n.get("pitch", 0)))
    starts = [n["start"] for n in notes_sorted]
    to_remove = set()
    
    # 共鳴パターン定義: (共鳴ピッチ, トリガーピッチのリスト)
    # E3(52) は E2(40) の共鳴
    # G3(55) は E2(40) または B3(59) の共鳴
    RESONANCE_PATTERNS = {
        52: [40],           # E3 ← E2
        55: [40, 59],       # G3 ← E2, B3
    }
    
    # E2が頻出する場合、E3は全てオクターブ共鳴として除去
    # （E2開放弦は非常に長くサステインし、E3を常に共鳴させる）
    e2_count = sum(1 for n in notes_sorted if n.get("pitch", 0) == 40)
    e3_global_remove = e2_count > 10  # E2が10回以上なら全E3を共鳴扱い
    
    for i, note in enumerate(notes_sorted):
        pitch = note.get("pitch", 0)
        if pitch not in RESONANCE_PATTERNS:
            continue
        
        t = note["start"]
        triggers = RESONANCE_PATTERNS[pitch]
        
        # E3 + E2が頻出 → 時間窓なしで全て共鳴として除去
        if pitch == 52 and e3_global_remove:
            to_remove.add(i)
            continue
        
        # G3等は従来通り時間窓ベースで判定
        # トリガーピッチが近傍に存在するか確認
        # E2開放弦は2-3秒サステインするため、広い範囲で検索
        lo = bisect_right(starts, t - 2.0)  # 2.0秒前まで検索
        hi = bisect_right(starts, t + 0.1)
        
        has_trigger = False
        for j in range(lo, hi):
            if j == i or j in to_remove:
                continue
            other = notes_sorted[j]
            if other.get("pitch", 0) in triggers:
                has_trigger = True
                break
        
        if not has_trigger:
            continue
        
        # トリガーが存在 → 共鳴の可能性あり
        # メロディ文脈があれば保護（実際に弾いた音）
        has_melody = False
        for j in range(lo, hi):
            if j == i or j in to_remove:
                continue
            other = notes_sorted[j]
            op = other.get("pitch", 0)
            interval = abs(op - pitch)
            # ステップ関係(1-5半音)で、トリガーでも共鳴候補でもなければメロディ
            all_resonance_pitches = set(RESONANCE_PATTERNS.keys())
            if 1 <= interval <= 5 and op not in triggers and op != pitch and op not in all_resonance_pitches:
                has_melody = True
                break
        
        if has_melody:
            continue
        
        to_remove.add(i)
    
    if to_remove:
        filtered = [n for i, n in enumerate(notes_sorted) if i not in to_remove]
        logger.info(f"[open_resonance] Removed {len(to_remove)} sympathetic resonance notes")
        return filtered
    
    return notes


def filter_max_per_beat(notes: List[Dict], beats: Optional[List[float]] = None,
                        bpm: float = 93.7, max_notes: int = 6) -> List[Dict]:
    """
    1拍あたりの最大ノート数制限。
    各拍で最大max_notes個のノートのみ残す。
    velocityの高いノートを優先。
    
    Args:
        notes: ノートリスト
        beats: ビート位置のリスト（秒）
        bpm: BPM（beatsがない場合に使用）
        max_notes: 1拍あたりの最大ノート数（ギターは6弦なので6が上限）
    """
    if not notes:
        return notes
    
    # ビートグリッドを作成
    if beats and len(beats) > 1:
        beat_edges = beats
    else:
        beat_duration = 60.0 / bpm
        max_time = max(n["start"] for n in notes) + 1
        beat_edges = [i * beat_duration for i in range(int(max_time / beat_duration) + 2)]
    
    # 各拍にノートを割り当て (bisectで高速化)
    beat_groups = {}
    for n in notes:
        beat_idx = max(0, bisect_right(beat_edges, n["start"]) - 1)
        
        if beat_idx not in beat_groups:
            beat_groups[beat_idx] = []
        beat_groups[beat_idx].append(n)
    
    # 各拍でmax_notes個を超えるノートを除去
    result = []
    removed = 0
    for beat_idx in sorted(beat_groups.keys()):
        group = beat_groups[beat_idx]
        if len(group) <= max_notes:
            result.extend(group)
        else:
            # 弦の多様性を優先: 各弦1音、velocityで選択
            by_string = {}
            for n in group:
                s = n.get("string", 0)
                if s not in by_string or n.get("velocity", 0) > by_string[s].get("velocity", 0):
                    by_string[s] = n
            
            selected = list(by_string.values())[:max_notes]
            result.extend(selected)
            removed += len(group) - len(selected)
    
    if removed:
        logger.info(f"[max_per_beat] Removed {removed} excess notes (limit: {max_notes}/beat)")
    return sorted(result, key=lambda n: (n["start"], n.get("pitch", 0)))


def filter_bass_octave_harmonics(notes: List[Dict], 
                                 bass_threshold: int = 52,
                                 time_window: float = 0.15) -> List[Dict]:
    """
    ベース音のオクターブ倍音フィルタ（v2強化版）。
    低音（E2=40等）のオクターブ上（E3=52等）に出現するファントムノートを除去。
    
    v2改善点:
    - ベース音との共起関係を活用: ベース音のvelocityとの比較で判定
    - time_windowを0.3秒に拡大してアルペジオのベース音もカバー
    
    Args:
        notes: ノートリスト
        bass_threshold: ベース音とみなす最大ピッチ（デフォルト52=E3)
        time_window: ベース音との時間窓（秒）
    """
    if not notes:
        return notes
    
    notes_sorted = sorted(notes, key=lambda n: n["start"])
    
    # ベース音を収集
    bass_notes = [n for n in notes_sorted if n.get("pitch", 127) <= bass_threshold]
    if not bass_notes:
        return notes
    
    # ベース音のオクターブ上ピッチマップ: {ピッチ: [(時間, velocity, 元ピッチ), ...]}
    octave_map = {}
    for bn in bass_notes:
        bp = bn.get("pitch", 0)
        bv = bn.get("velocity", 0.5)
        bt = bn["start"]
        
        # オクターブ上のみ (+12半音)
        oct_up = bp + 12
        if oct_up not in octave_map:
            octave_map[oct_up] = []
        octave_map[oct_up].append((bt, bv, bp))
    
    keep = []
    removed = 0
    extended_window = 0.3  # ベース共起検出はやや広めのウィンドウ
    
    for n in notes_sorted:
        pitch = n.get("pitch", 0)
        t = n["start"]
        v = n.get("velocity", 0.5)
        dur = n.get("end", t) - t
        
        # このノートがベース音のオクターブ上ピッチか？
        if pitch in octave_map:
            bass_info = octave_map[pitch]
            # 近傍のベース音を探す
            nearby_bass = [(bt, bv, bp) for bt, bv, bp in bass_info 
                           if abs(t - bt) < extended_window]
            
            if nearby_bass:
                # ベース音の最大velocityと比較
                max_bass_vel = max(bv for _, bv, _ in nearby_bass)
                
                # ベース音のvelocity×1.1以下なら倍音として除去
                if v <= max_bass_vel * 1.1:
                    removed += 1
                    continue
            else:
                # ベース音が見つからない場合: 従来の緩い条件
                if v < 0.6 and dur < 0.2:
                    removed += 1
                    continue
        
        keep.append(n)
    
    if removed:
        logger.info(f"[bass_octave_harmonics] Removed {removed} bass octave phantom notes")
    return keep


def apply_all_filters(notes: List[Dict], 
                      velocity_threshold: float = 0.3,
                      min_duration_sec: float = 0.1,
                      enable_harmonics: bool = True,
                      enable_position_window: bool = True,
                      window_size: int = 7,
                      max_notes_per_beat: int = 6,
                       beats: Optional[List[float]] = None,
                      bpm: float = 93.7) -> Dict:
    """
    全フィルタを順番に適用する。
    
    Returns:
        dict: {"notes": filtered_notes, "stats": {filter_name: removed_count}}
    """
    stats = {}
    original_count = len(notes)
    
    # 1. close duplicate除去（最も効果的）
    pre = len(notes)
    notes = filter_close_duplicates(notes, time_tolerance=0.04) # 100ms -> 40ms to protect tremolo
    stats["close_duplicates"] = pre - len(notes)
    
    # 1.5. 単独検出ノイズ除去 — アンサンブル段階で min_models_for_accept により
    # 既にフィルタ済みのため、ここでの再フィルタは省略。
    # note_filterは重複除去とvelocity/durationベースのフィルタに集中。
    
    # 2. velocity閾値フィルタ
    pre = len(notes)
    notes = filter_by_velocity(notes, threshold=velocity_threshold)
    stats["velocity"] = pre - len(notes)
    
    # 3. 最小音価フィルタ（100msに引き上げ）
    pre = len(notes)
    notes = filter_by_min_duration(notes, min_duration_sec=min_duration_sec)
    stats["min_duration"] = pre - len(notes)
    
    # 4. 開放弦共鳴ノート除去
    pre = len(notes)
    notes = filter_open_string_resonance(notes)
    stats["open_resonance"] = pre - len(notes)
    
    # 5. 倍音フィルタ（ハーモニクス保持版）
    if enable_harmonics:
        pre = len(notes)
        notes = filter_harmonics(notes, time_tolerance=0.2, velocity_ratio=0.7)
        stats["harmonics"] = pre - len(notes)
    
    # 5.5. ベースオクターブ倍音フィルタ
    pre = len(notes)
    notes = filter_bass_octave_harmonics(notes)
    stats["bass_octave"] = pre - len(notes)
    
    # 6. 1拍あたり最大ノート数制限
    pre = len(notes)
    notes = filter_max_per_beat(notes, beats=beats, bpm=bpm, 
                                 max_notes=max_notes_per_beat)
    stats["max_per_beat"] = pre - len(notes)
    
    # 7. ポジション固定ウィンドウ（ノート除去ではなくリマップ）
    if enable_position_window:
        notes = filter_position_window(notes, window_size=window_size)
        stats["position_window"] = "applied"
    
    stats["original"] = original_count
    stats["filtered"] = len(notes)
    stats["removed_total"] = original_count - len(notes)
    stats["reduction_pct"] = round((1 - len(notes) / max(original_count, 1)) * 100, 1)
    
    logger.info(
        f"[note_filter] {original_count} → {len(notes)} notes "
        f"(-{stats['removed_total']}, -{stats['reduction_pct']}%)"
    )
    
    return {"notes": notes, "stats": stats}
