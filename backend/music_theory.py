"""
music_theory.py — 音楽理論ベースのノート補正エンジン
====================================================
検出されたノートを音楽理論の観点から検証・補正する。

基礎概念:
  - キー (調): 曲全体で使われるスケールの基盤
  - スケール (音階): キーに基づく許容ピッチクラスの集合
  - コード構成音: 現在のコードに含まれる音 → 拍頭で優先
  - ギター音域: 各弦の物理的なピッチ範囲 (E2=40 ~ E6=88)
  - オクターブ整合性: 高フレットより低フレットの同名音を優先

パイプライン位置:
  MoE検出 → [music_theory補正] → 弦割り当て → MusicXML生成
"""

from typing import List, Dict, Optional, Tuple
from collections import Counter
import math


# =============================================================================
# 1. 基礎定数
# =============================================================================

# ピッチクラス名 → 番号
PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PC_MAP = {name: i for i, name in enumerate(PC_NAMES)}
PC_MAP.update({'Db': 1, 'Eb': 3, 'Fb': 4, 'Gb': 6, 'Ab': 8, 'Bb': 10, 'Cb': 11})

# スケール定義 (ルートからの半音数)
SCALES = {
    'major':          [0, 2, 4, 5, 7, 9, 11],
    'natural_minor':  [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor':  [0, 2, 3, 5, 7, 9, 11],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
}

# コードクオリティ → 構成音 (ルートからの半音数)
CHORD_TONES = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    '7':     [0, 4, 7, 10],
    'm7':    [0, 3, 7, 10],
    'maj7':  [0, 4, 7, 11],
    'dim':   [0, 3, 6],
    'dim7':  [0, 3, 6, 9],
    'aug':   [0, 4, 8],
    'sus4':  [0, 5, 7],
    'sus2':  [0, 2, 7],
    'm7b5':  [0, 3, 6, 10],
    '6':     [0, 4, 7, 9],
    'm6':    [0, 3, 7, 9],
    'add9':  [0, 2, 4, 7],
    '9':     [0, 2, 4, 7, 10],
}

# ギター標準チューニング
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
MAX_FRET = 19

# ギターの物理的音域
GUITAR_MIN_PITCH = 40   # E2 (6弦開放)
GUITAR_MAX_PITCH = 83   # B5 (1弦19フレット)


# =============================================================================
# 2. コード解析
# =============================================================================

def parse_chord(chord_str: str) -> Tuple[int, str, List[int]]:
    """
    コード名を解析してルートPC、クオリティ、構成音PCリストを返す。
    
    例: 'Am7' -> (9, 'm7', [9, 0, 4, 7])
        'C'   -> (0, 'major', [0, 4, 7])
    
    Returns:
        (root_pc, quality, tone_pcs) or (-1, '', []) for N/A
    """
    if not chord_str or chord_str in ('N', 'X', 'NC', 'N.C.', 'silence'):
        return (-1, '', [])
    
    s = chord_str.strip()
    
    # ルート音
    if s[0] not in PC_MAP:
        return (-1, '', [])
    
    root_pc = PC_MAP[s[0]]
    idx = 1
    
    if idx < len(s) and s[idx] in ('#', 'b'):
        modifier = s[idx]
        root_pc = (root_pc + (1 if modifier == '#' else -1)) % 12
        idx += 1
    
    # スラッシュコード (C/E → ルートCのまま)
    quality_str = s[idx:].split('/')[0].lower()
    
    # クオリティ判定
    if quality_str in ('', 'maj'):
        quality = 'major'
    elif quality_str in ('m', 'min', 'mi', '-'):
        quality = 'minor'
    elif quality_str in ('7', 'dom7'):
        quality = '7'
    elif quality_str in ('m7', 'min7', '-7'):
        quality = 'm7'
    elif quality_str in ('maj7', 'ma7', 'M7'):
        quality = 'maj7'
    elif quality_str in ('dim', 'o', 'mb5'):
        quality = 'dim'
    elif quality_str in ('dim7', 'o7'):
        quality = 'dim7'
    elif quality_str in ('aug', '+'):
        quality = 'aug'
    elif quality_str in ('sus4', 'sus'):
        quality = 'sus4'
    elif quality_str in ('sus2',):
        quality = 'sus2'
    elif quality_str in ('m7b5', 'ø', 'ø7'):
        quality = 'm7b5'
    elif quality_str in ('6',):
        quality = '6'
    elif quality_str in ('m6',):
        quality = 'm6'
    elif quality_str in ('add9',):
        quality = 'add9'
    elif quality_str in ('9',):
        quality = '9'
    else:
        quality = 'major'  # fallback
    
    intervals = CHORD_TONES.get(quality, [0, 4, 7])
    tone_pcs = [(root_pc + iv) % 12 for iv in intervals]
    
    return (root_pc, quality, tone_pcs)


# =============================================================================
# 3. キー推定
# =============================================================================

def estimate_key(chords: List[Dict]) -> Tuple[int, str]:
    """
    コード進行からキーを推定する (Krumhansl-Schmuckler 簡易版)。
    
    Args:
        chords: [{'chord': 'Am', 'start': 0.0, 'end': 2.0}, ...]
    
    Returns:
        (key_root_pc, scale_type) — 例: (4, 'natural_minor') = Em
    """
    if not chords:
        return (0, 'major')
    
    # コードルートの出現頻度を重み付きで集計
    root_weights = Counter()
    for c in chords:
        root_pc, quality, _ = parse_chord(c.get('chord', ''))
        if root_pc < 0:
            continue
        dur = c.get('end', 0) - c.get('start', 0)
        dur = max(dur, 0.1)
        root_weights[root_pc] += dur
        
        # マイナーコードならマイナーキーの証拠
        if quality in ('minor', 'm7'):
            root_weights[(root_pc, 'minor')] = root_weights.get((root_pc, 'minor'), 0) + dur
    
    if not root_weights:
        return (0, 'major')
    
    # 最もスコアの高いキーを選択
    best_key = (0, 'major')
    best_score = -1
    
    for key_root in range(12):
        for scale_name, scale_intervals in [('major', SCALES['major']), 
                                              ('natural_minor', SCALES['natural_minor'])]:
            scale_pcs = set((key_root + iv) % 12 for iv in scale_intervals)
            score = 0
            
            for pc, weight in root_weights.items():
                if isinstance(pc, tuple):
                    continue  # minor evidence entries
                if pc in scale_pcs:
                    score += weight
                    # トニック(I)のルートにボーナス
                    if pc == key_root:
                        score += weight * 0.5
                    # ドミナント(V)のルートにボーナス
                    if pc == (key_root + 7) % 12:
                        score += weight * 0.3
            
            # マイナーキーの証拠を加味
            if scale_name == 'natural_minor':
                minor_ev = root_weights.get((key_root, 'minor'), 0)
                score += minor_ev * 0.5
            
            if score > best_score:
                best_score = score
                best_key = (key_root, scale_name)
    
    return best_key


# =============================================================================
# 4. ノート検証・補正
# =============================================================================

def get_scale_pcs(key_root: int, scale_type: str) -> set:
    """キーのスケール構成音PCセットを返す。"""
    intervals = SCALES.get(scale_type, SCALES['major'])
    return set((key_root + iv) % 12 for iv in intervals)


def find_chord_at_time(chords: List[Dict], time: float) -> Tuple[int, str, List[int]]:
    """指定時刻のコードを返す。"""
    for c in chords:
        if c.get('start', 0) <= time < c.get('end', float('inf')):
            return parse_chord(c.get('chord', ''))
    return (-1, '', [])


def correct_note_pitch(note: Dict, chords: List[Dict], 
                        key_root: int, scale_pcs: set,
                        tuning: List[int] = None) -> Dict:
    """
    音楽理論に基づいてノートのピッチを検証・補正する。
    
    補正ルール (優先度順):
    1. ギター物理音域外 → 除外フラグ
    2. コード構成音チェック → 構成音なら信頼度UP
    3. スケール外ノート → 最近傍スケールトーンに補正
    4. 高フレット補正 → 同じピッチクラスの低フレット代替を検討
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    pitch = note['pitch']
    pc = pitch % 12
    
    # --- ルール1: 物理音域チェック ---
    if pitch < GUITAR_MIN_PITCH or pitch > GUITAR_MAX_PITCH:
        note['_theory_flag'] = 'out_of_range'
        return note
    
    # --- 現在のコード取得 ---
    chord_root, chord_quality, chord_tone_pcs = find_chord_at_time(chords, note['start'])
    
    # --- ルール2: コード構成音チェック ---
    is_chord_tone = pc in chord_tone_pcs if chord_tone_pcs else False
    is_scale_tone = pc in scale_pcs
    
    note['_is_chord_tone'] = is_chord_tone
    note['_is_scale_tone'] = is_scale_tone
    
    if is_chord_tone:
        # コード構成音 → 信頼度高い。ただしオクターブの妥当性は別途チェック
        note['_theory_flag'] = 'chord_tone'
        return note
    
    if is_scale_tone:
        # スケール内だがコード構成音でない → パッシングトーン/テンション
        note['_theory_flag'] = 'scale_tone'
        return note
    
    # --- ルール3: スケール外ノートの補正 ---
    # 最近傍のスケールトーン (コード構成音優先) を探す
    best_correction = None
    best_dist = float('inf')
    
    for offset in range(-2, 3):  # ±2半音以内
        candidate_pitch = pitch + offset
        candidate_pc = candidate_pitch % 12
        
        if offset == 0:
            continue
        
        # ギター音域内かチェック
        if candidate_pitch < GUITAR_MIN_PITCH or candidate_pitch > GUITAR_MAX_PITCH:
            continue
        
        dist = abs(offset)
        
        # コード構成音への補正を強く優先
        if candidate_pc in chord_tone_pcs:
            effective_dist = dist * 0.5  # コード構成音は距離を半分に
        elif candidate_pc in scale_pcs:
            effective_dist = dist
        else:
            continue  # スケール外への補正は不許可
        
        if effective_dist < best_dist:
            best_dist = effective_dist
            best_correction = candidate_pitch
    
    if best_correction is not None and best_dist <= 1.5:
        original_pitch = note['pitch']
        note['pitch'] = best_correction
        note['_theory_flag'] = 'corrected'
        note['_original_pitch'] = original_pitch
        note['_correction_reason'] = f"scale_snap: {original_pitch} -> {best_correction}"
    else:
        note['_theory_flag'] = 'non_scale'
    
    return note


def apply_music_theory_filter(notes: List[Dict], chords: List[Dict],
                                tuning: List[int] = None) -> List[Dict]:
    """
    音楽理論フィルタをノートリスト全体に適用する。
    
    Args:
        notes: 検出済みノートリスト [{'start', 'pitch', 'string', 'fret', ...}]
        chords: コード検出結果 [{'chord', 'start', 'end'}, ...]
        tuning: チューニング (デフォルト: standard)
    
    Returns:
        補正済みノートリスト
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    if not notes:
        return notes
    
    # --- キー推定 ---
    key_root, scale_type = estimate_key(chords)
    scale_pcs = get_scale_pcs(key_root, scale_type)
    key_name = f"{PC_NAMES[key_root]} {scale_type}"
    
    print(f"[music_theory] Key estimated: {key_name}")
    print(f"[music_theory] Scale PCs: {sorted(scale_pcs)}")
    print(f"[music_theory] Processing {len(notes)} notes with {len(chords)} chord regions")
    
    # --- 各ノートを検証・補正 ---
    corrected_count = 0
    removed_count = 0
    
    result = []
    for note in notes:
        note = correct_note_pitch(note, chords, key_root, scale_pcs, tuning)
        
        flag = note.get('_theory_flag', '')
        
        if flag == 'out_of_range':
            removed_count += 1
            continue  # 音域外は除去
        
        if flag == 'corrected':
            corrected_count += 1
        
        result.append(note)
    
    # --- 統計 (スケール補正) ---
    chord_tones = sum(1 for n in result if n.get('_is_chord_tone'))
    scale_tones = sum(1 for n in result if n.get('_is_scale_tone'))
    non_scale = sum(1 for n in result if n.get('_theory_flag') == 'non_scale')
    
    print(f"[music_theory] Scale filter: {len(result)} notes")
    print(f"  Chord tones: {chord_tones} ({100*chord_tones/max(len(result),1):.1f}%)")
    print(f"  Scale tones: {scale_tones} ({100*scale_tones/max(len(result),1):.1f}%)")
    print(f"  Non-scale:   {non_scale} ({100*non_scale/max(len(result),1):.1f}%)")
    print(f"  Scale corrected: {corrected_count}, Removed: {removed_count}")
    
    # =====================================================================
    # Phase 2: アルペジオ・パターン補完
    # =====================================================================
    # 繰り返しアルペジオで検出漏れしたメロディ音を補完する。
    # 原理: 同じコード内の拍が同じパターンを持つはずなので、
    #       ある拍で検出されたノートが別の拍で欠落していたら補完。
    #
    # 注意: インターバルフィルタは無効化。
    #       高フレット(7,12等)は正しい場合が多く、オクターブ補正は有害。
    
    SIMULTANEOUS_THRESHOLD = 0.03  # 同時発音の閾値 (秒)
    pattern_completions = 0
    
    # 時間順ソート
    result.sort(key=lambda n: (n['start'], -n['pitch']))
    
    # ビートグループ化: 同時発音ノートをグループにまとめる
    beat_groups = []  # [(start_time, [notes])]
    i = 0
    while i < len(result):
        group_start = result[i]['start']
        group = [result[i]]
        j = i + 1
        while j < len(result) and abs(result[j]['start'] - group_start) < SIMULTANEOUS_THRESHOLD:
            group.append(result[j])
            j += 1
        beat_groups.append((group_start, group))
        i = j
    
    # コード区間ごとにパターン分析
    if chords:
        for chord_info in chords:
            chord_start = chord_info.get('start', 0)
            chord_end = chord_info.get('end', 0)
            chord_name = chord_info.get('chord', '')
            _, _, chord_tone_pcs = parse_chord(chord_name)
            
            if not chord_tone_pcs:
                continue
            
            # このコード区間内のビートグループ
            chord_beats = [(t, g) for t, g in beat_groups 
                          if chord_start <= t < chord_end]
            
            if len(chord_beats) < 3:
                continue
            
            # 各ビートのピッチ集合を収集
            all_pitches_in_chord = set()
            for _, group in chord_beats:
                for n in group:
                    all_pitches_in_chord.add(n['pitch'])
            
            # 最高音（メロディ候補）を特定
            max_pitch = max(all_pitches_in_chord) if all_pitches_in_chord else 0
            
            # メロディ音がコード構成音かチェック
            if max_pitch % 12 not in chord_tone_pcs:
                continue
            
            # 各ビートでメロディ音が存在するか確認
            for beat_time, group in chord_beats:
                group_pitches = [n['pitch'] for n in group]
                group_max = max(group_pitches) if group_pitches else 0
                
                # メロディ音が欠落している場合 (最高音がオクターブ以上低い)
                if max_pitch - group_max >= 12:
                    # メロディ音を補完
                    # 元のノートの弦/フレットを推定
                    melody_fret = None
                    melody_string = None
                    
                    # 同じコード内でメロディ音が検出されたビートから弦/フレットを借用
                    for _, ref_group in chord_beats:
                        for rn in ref_group:
                            if rn['pitch'] == max_pitch:
                                melody_string = rn['string']
                                melody_fret = rn['fret']
                                break
                        if melody_string is not None:
                            break
                    
                    if melody_string is not None and melody_fret is not None:
                        new_note = {
                            'start': beat_time,
                            'end': beat_time + 0.2,
                            'pitch': max_pitch,
                            'string': melody_string,
                            'fret': melody_fret,
                            'velocity': 0.7,
                            'source': 'pattern_completion',
                            '_theory_flag': 'pattern_completed',
                            '_correction_reason': f"melody_fill: {max_pitch} from chord {chord_name}",
                        }
                        result.append(new_note)
                        pattern_completions += 1
    
    # 再ソート
    result.sort(key=lambda n: (n['start'], -n['pitch']))
    
    print(f"[music_theory] Pattern completions: {pattern_completions}")
    
    # =====================================================================
    # Phase 3: コードボイシングによる弦割り当て修正
    # =====================================================================
    # コードの標準ボイシングを使って、同じピッチを出せる複数の弦/フレット
    # 候補から最も自然な選択肢を選ぶ。
    # 例: B3(59) → 3弦4フレット vs 2弦0フレット → Em開放では2弦0が正解
    
    voicing_corrections = 0
    
    # ボイシングDB読み込み
    chord_forms_db = []
    try:
        import os
        db_path = os.path.join(os.path.dirname(__file__), "chord_forms_db.json")
        if os.path.exists(db_path):
            import json as _json
            with open(db_path, encoding='utf-8') as f:
                chord_forms_db = _json.load(f)
    except Exception:
        pass
    
    if chord_forms_db and chords:
        # コード名→フォームのルックアップ構築
        forms_lookup = {}
        for form in chord_forms_db:
            ch = form['chord']
            if ch not in forms_lookup:
                forms_lookup[ch] = []
            forms_lookup[ch].append(form)
        
        for note in result:
            pitch = note['pitch']
            s = note.get('string', 0)
            f = note.get('fret', 0)
            
            # このノートの時刻のコード
            chord_root, chord_quality, chord_tone_pcs = find_chord_at_time(
                chords, note['start'])
            if chord_root < 0:
                continue
            
            # ピッチがコード構成音でなければスキップ
            if pitch % 12 not in chord_tone_pcs:
                continue
            
            # コードのフォームを取得
            chord_name_candidates = []
            for c in chords:
                if c.get('start', 0) <= note['start'] < c.get('end', float('inf')):
                    chord_name_candidates.append(c.get('chord', ''))
                    break
            
            if not chord_name_candidates:
                continue
            
            chord_name = chord_name_candidates[0]
            forms = forms_lookup.get(chord_name, [])
            
            # 開放ポジション(position=0)のフォームを優先
            open_forms = [fm for fm in forms if fm.get('position', 0) == 0 
                         and not fm.get('partial')]
            if not open_forms:
                open_forms = [fm for fm in forms if not fm.get('partial')]
            if not open_forms:
                continue
            
            # 最適なフォームからこのピッチの弦/フレットを探す
            for form in open_forms:
                form_notes = form.get('notes', [])
                form_frets = form.get('frets', [])
                
                for si in range(6):
                    if si < len(form_notes) and form_notes[si] == pitch:
                        correct_string = 6 - si  # DB: index 0=6弦, SoloTab: string 1=1弦
                        correct_fret = form_frets[si] if si < len(form_frets) else -1
                        
                        if correct_fret >= 0 and (correct_string != s or correct_fret != f):
                            # 開放弦を優先（左手不要）
                            if correct_fret == 0 or correct_fret < f:
                                note['string'] = correct_string
                                note['fret'] = correct_fret
                                note['_voicing_corrected'] = True
                                voicing_corrections += 1
                        break
    
    print(f"[music_theory] Voicing corrections: {voicing_corrections}")
    
    # =====================================================================
    # Phase 4: 声部分離 (メロディ / 内声 / ベース)
    # =====================================================================
    # 同時発音ノートをメロディ(最高音)、ベース(最低音)、内声(中間)に分類。
    # TAB表記やアーティキュレーションの判断基盤。
    
    SIMULTANEOUS_THRESHOLD_V = 0.03
    result.sort(key=lambda n: (n['start'], -n['pitch']))
    
    i = 0
    while i < len(result):
        group_start = result[i]['start']
        group = [i]
        j = i + 1
        while j < len(result) and abs(result[j]['start'] - group_start) < SIMULTANEOUS_THRESHOLD_V:
            group.append(j)
            j += 1
        
        if len(group) >= 3:
            pitches = [(result[idx]['pitch'], idx) for idx in group]
            pitches.sort()
            result[pitches[-1][1]]['_voice'] = 'melody'
            result[pitches[0][1]]['_voice'] = 'bass'
            for _, idx in pitches[1:-1]:
                result[idx]['_voice'] = 'inner'
        elif len(group) == 2:
            pitches = [(result[idx]['pitch'], idx) for idx in group]
            pitches.sort()
            result[pitches[-1][1]]['_voice'] = 'melody'
            result[pitches[0][1]]['_voice'] = 'bass'
        else:
            result[i]['_voice'] = 'melody'
        
        i = j
    
    voice_counts = {}
    for n in result:
        v = n.get('_voice', 'unknown')
        voice_counts[v] = voice_counts.get(v, 0) + 1
    print(f"[music_theory] Voices: {voice_counts}")
    
    # =====================================================================
    # Phase 5: ギターポジション推定
    # =====================================================================
    # コード区間ごとに使用ポジション（フレット帯域）を推定し記録。
    # ポジション外のフレットを持つノートは将来的に修正候補。
    
    if chords:
        for chord_info in chords:
            cs = chord_info.get('start', 0)
            ce = chord_info.get('end', 0)
            
            chord_notes = [n for n in result if cs <= n['start'] < ce]
            if not chord_notes:
                continue
            
            frets = [n['fret'] for n in chord_notes if n.get('fret', 0) > 0]
            if frets:
                pos_min = min(frets)
                pos_max = max(frets)
                pos_center = sum(frets) / len(frets)
                for n in chord_notes:
                    n['_position'] = round(pos_center, 1)
                    n['_position_range'] = (pos_min, pos_max)
    
    # =====================================================================
    # Phase 6: リズム構造検出 (3連符)
    # =====================================================================
    # ノート間の時間間隔から3連符パターンを検出し記録。
    
    triplet_count = 0
    if len(result) >= 3:
        for i in range(len(result) - 2):
            t0 = result[i]['start']
            t1 = result[i+1]['start']
            t2 = result[i+2]['start']
            
            d1 = t1 - t0
            d2 = t2 - t1
            
            if d1 > 0.05 and d2 > 0.05:  # 極端に短い間隔は除外
                ratio = d1 / d2 if d2 > 0 else 0
                # 3連符: 等間隔 (ratio ≈ 1.0)
                if 0.8 <= ratio <= 1.2:
                    # 3連の1拍分がビート間隔の1/3か確認
                    # BPM 90, 3/4: beat=0.667s, triplet=0.222s
                    if 0.1 <= d1 <= 0.35:
                        result[i]['_rhythm'] = 'triplet'
                        result[i+1]['_rhythm'] = 'triplet'
                        result[i+2]['_rhythm'] = 'triplet'
                        triplet_count += 1
    
    print(f"[music_theory] Triplet groups detected: {triplet_count}")
    
    total_corrected = corrected_count + pattern_completions + voicing_corrections
    print(f"[music_theory] Total corrections: {total_corrected} / {len(result)} notes")
    
    return result


# =============================================================================
# 5. テスト
# =============================================================================

if __name__ == "__main__":
    # 禁じられた遊び冒頭のテスト
    test_chords = [
        {'chord': 'Em', 'start': 0.0, 'end': 4.0},
        {'chord': 'Am', 'start': 4.0, 'end': 8.0},
    ]
    
    test_notes = [
        {'start': 1.9, 'pitch': 40, 'string': 6, 'fret': 0},   # E2 → Em構成音 ✓
        {'start': 1.9, 'pitch': 71, 'string': 1, 'fret': 7},   # B4 → Em構成音だが高い
        {'start': 2.1, 'pitch': 59, 'string': 3, 'fret': 4},   # B3 → Em構成音 ✓
        {'start': 2.3, 'pitch': 55, 'string': 3, 'fret': 0},   # G3 → Em構成音 ✓
        {'start': 2.5, 'pitch': 59, 'string': 3, 'fret': 4},   # B3 → Em構成音 ✓
        {'start': 4.5, 'pitch': 62, 'string': 1, 'fret': -2},  # D4 → Am外? テスト用
    ]
    
    result = apply_music_theory_filter(test_notes, test_chords)
    
    print("\n=== Results ===")
    for n in result:
        flag = n.get('_theory_flag', '')
        corr = n.get('_correction_reason', '')
        print(f"  t={n['start']:.1f} pitch={n['pitch']} ({PC_NAMES[n['pitch']%12]}) "
              f"flag={flag} {corr}")
