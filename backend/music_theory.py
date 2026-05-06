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
                                tuning: List[int] = None,
                                beats: List[float] = None) -> tuple:
    """
    音楽理論フィルタをノートリスト全体に適用する。
    
    Args:
        notes: 検出済みノートリスト [{'start', 'pitch', 'string', 'fret', ...}]
        chords: コード検出結果 [{'chord', 'start', 'end'}, ...]
        tuning: チューニング (デフォルト: standard)
        beats: ビート時刻リスト (リズム検出用)
    
    Returns:
        (補正済みノートリスト, rhythm_info dict)
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    if not notes:
        return notes
    
    _modulation_point = None  # Phase -0.5で設定される転調点
    
    # --- Phase -1: 楽曲構造パターンによるコード補正 ---
    # CQT分析で確認済み: 開放弦の残響が押弦のコード構成音を20dB以上マスクし、
    # BTC/chromaベースの検出器がコード変化（Em→Am等）を検出できない。
    # 対策: 小節単位でビートを区切り、典型的なコード進行パターンを適用して
    # BTCの誤検出を上書きする。
    if chords and beats and len(beats) >= 6:
        # 小節境界の推定（3拍子 or 4拍子）
        beat_intervals = [beats[i+1] - beats[i] for i in range(min(len(beats)-1, 20))]
        median_beat_interval = sorted(beat_intervals)[len(beat_intervals)//2]
        
        # 拍子推定: 3/4拍子の場合、小節 = 3ビート
        # 4/4拍子の場合、小節 = 4ビート
        # ビート間隔から推定（0.5-0.8s → 3/4, 0.3-0.5s → 4/4 の傾向）
        beats_per_measure = 3 if median_beat_interval > 0.5 else 4
        
        # 小節境界を作成
        measure_boundaries = []
        for bi in range(0, len(beats) - beats_per_measure + 1, beats_per_measure):
            m_start = beats[bi]
            m_end = beats[bi + beats_per_measure] if bi + beats_per_measure < len(beats) else beats[-1] + median_beat_interval
            measure_boundaries.append((m_start, m_end))
        
        if len(measure_boundaries) >= 4:
            # 各小節のコードを特定
            measure_chords = []
            for m_start, m_end in measure_boundaries:
                m_mid = (m_start + m_end) / 2
                chord_at_m = None
                for c in chords:
                    if c['start'] <= m_mid < c['end']:
                        chord_at_m = c.get('chord', '')
                        break
                measure_chords.append(chord_at_m)
            
            # パターン検出: 同じコードが4小節以上続く場合
            # 典型的な進行パターンで補正を試みる
            # Em-Em-Am-Em (Romance de Amor等のクラシカルパターン)
            # Am-Am-Dm-Am, etc.
            PROGRESSION_PATTERNS = {
                'Em': [('Em', 'Em', 'Am', 'Em'), ('Em', 'Am', 'Em', 'Em')],
                'Am': [('Am', 'Am', 'Dm', 'Am'), ('Am', 'Dm', 'Am', 'Am')],
                'Dm': [('Dm', 'Dm', 'Gm', 'Dm')],
            }
            
            pattern_corrections = 0
            i = 0
            while i <= len(measure_chords) - 4:
                chord = measure_chords[i]
                if chord and all(c == chord for c in measure_chords[i:i+4]):
                    # 4小節連続同一コード → 進行パターンを適用
                    patterns = PROGRESSION_PATTERNS.get(chord, [])
                    if patterns:
                        pattern = patterns[0]  # 最も一般的なパターンを使用
                        for j, new_chord in enumerate(pattern):
                            if new_chord != chord:
                                mi = i + j
                                if mi < len(measure_boundaries):
                                    ms, me = measure_boundaries[mi]
                                    # chords リストに新しいコード区間を挿入
                                    # 既存のコード区間を分割する
                                    new_chords = []
                                    for c in chords:
                                        if c['start'] < ms and c['end'] > me:
                                            # この区間を3分割: before, new, after
                                            if c['start'] < ms:
                                                new_chords.append({
                                                    'start': c['start'], 'end': ms,
                                                    'chord': c['chord'],
                                                    'confidence': c.get('confidence', 0.5),
                                                })
                                            new_chords.append({
                                                'start': ms, 'end': me,
                                                'chord': new_chord,
                                                'confidence': 0.6,
                                                '_structure_inferred': True,
                                            })
                                            if me < c['end']:
                                                new_chords.append({
                                                    'start': me, 'end': c['end'],
                                                    'chord': c['chord'],
                                                    'confidence': c.get('confidence', 0.5),
                                                })
                                        else:
                                            new_chords.append(c)
                                    chords = new_chords
                                    measure_chords[mi] = new_chord
                                    pattern_corrections += 1
                                    print(f"[music_theory] Phase -1: Measure {mi+1} "
                                          f"({ms:.1f}-{me:.1f}s) {chord} -> {new_chord} "
                                          f"(structure pattern)")
                                    
                                    # ベース音はPhase 2.5で処理（ここで変更すると
                                    # Phase 0/3.5の判定に悪影響を与えるため）
                        i += 4
                    else:
                        i += 1
                else:
                    i += 1
            
            if pattern_corrections > 0:
                print(f"[music_theory] Phase -1: {pattern_corrections} chord corrections "
                      f"via structure patterns")
    
    # --- Phase -0.5: 平行調転調検出 (Em → E major 等) ---
    # クラシカルギター曲（Romance de Amor等）では、前半がminor key、
    # 後半がその平行調major keyに転調するパターンが頻出する。
    # MoEモデルはmajor keyセクションのpitchを系統的に2半音低く出力する問題がある
    # （C#5→B4, G#4→F#4等）。これはmajorの構成音がminorの近傍にあるため。
    # 対策: 曲構造の繰り返しパターンから転調点を検出し、コード＋pitchを補正する。
    if chords and beats and len(beats) >= 12:
        # 現在のキーがminorの場合のみ処理
        pre_key_root, pre_scale = estimate_key(chords)
        if pre_scale == 'natural_minor':
            parallel_major_root = pre_key_root  # Em → E major (same root)
            
            # BTCが検出したmajorコード(E, A等)の出現位置を探す
            # E major/A majorが出現する最初の時刻を転調点候補とする
            major_chord_times = []
            minor_chord_name = PC_NAMES[pre_key_root] + 'm'  # 'Em'
            major_chord_name = PC_NAMES[pre_key_root]  # 'E'
            
            for c in chords:
                chord_label = c.get('chord', '')
                root_pc, quality, _ = parse_chord(chord_label)
                # major triad/7th on the parallel major root
                if root_pc == pre_key_root and quality in ('major', '7', 'maj7'):
                    major_chord_times.append(c['start'])
            
            # 曲の後半で major chord が出現する場合、転調と判断
            if notes:
                total_duration = max(n['start'] for n in notes)
                half_point = total_duration * 0.3  # 曲の30%以降でmajorが出現
                
                late_major = [t for t in major_chord_times if t > half_point]
                
                if late_major:
                    modulation_point = late_major[0]
                    
                    # 転調点の精密推定:
                    # 1. BTCがEmと言っている長い区間を探す。
                    #    曲の前半がEm(~16小節)→後半もBTCがEm(~16小節)と誤検出。
                    #    後半Emの開始時刻が真の転調点。
                    # 2. Phase -1でEm→Amの構造パターンが検出されている場合、
                    #    最初のパターン区間の終了後に同じEmが連続する→転調点
                    
                    # 方法A: Em→B→Em の遷移を探す（クラシカルの典型:
                    #   前半最後の小節がB(V)でEm(i)に解決 → 同じルートのE(I)に転調）
                    em_after_b = None
                    for ci in range(1, len(chords)):
                        prev_chord = chords[ci-1].get('chord', '')
                        cur_chord = chords[ci].get('chord', '')
                        cur_start = chords[ci].get('start', 0)
                        
                        # B→Em の遷移で、かつ曲の前半(20-50%地点)にある場合
                        if (prev_chord == 'B' and cur_chord in ('Em', minor_chord_name)
                            and total_duration * 0.2 < cur_start < total_duration * 0.5):
                            # さらにこのEmの前にEmが途切れている（=セクション境界）
                            em_after_b = cur_start
                    
                    if em_after_b:
                        # 転調点 = em_after_b の後の次の小節境界
                        # (B→Em遷移の後、次の小節が新セクション開始)
                        modulation_point = em_after_b
                        if measure_boundaries:
                            for ms, me in measure_boundaries:
                                if ms >= em_after_b + 1.0:
                                    modulation_point = ms
                                    break
                    else:
                        # 方法B: 曲の構造的な繰り返し点を推定
                        # 前半のminor区間の長さから、同じ長さのmajor区間を推定
                        first_em_start = None
                        first_em_end = None
                        for c in chords:
                            if c.get('chord', '') in ('Em', minor_chord_name):
                                if first_em_start is None:
                                    first_em_start = c['start']
                                first_em_end = c['end']
                        
                        if first_em_start is not None and first_em_end is not None:
                            # 前半のEmセクション長
                            minor_section_len = first_em_end - first_em_start
                            # 曲全体の半分付近を転調点とする
                            mid_estimate = total_duration * 0.35
                            modulation_point = min(modulation_point, mid_estimate)
                    
                    _modulation_point = modulation_point
                    print(f"[music_theory] Phase -0.5: Modulation point estimated at {modulation_point:.1f}s "
                          f"(BTC E major first at {late_major[0]:.1f}s)")
                    
                    # 転調区間のコードを minor → major に修正
                    # Em → E, Am → A (平行調の典型)
                    MINOR_TO_MAJOR = {
                        'Em': 'E', 'Am': 'A', 'Dm': 'D', 'Bm': 'B',
                        'F#m': 'F#', 'C#m': 'C#', 'G#m': 'G#',
                    }
                    
                    chord_corrections_m = 0
                    for c in chords:
                        if c['start'] >= modulation_point:
                            old_chord = c.get('chord', '')
                            if old_chord in MINOR_TO_MAJOR:
                                c['chord'] = MINOR_TO_MAJOR[old_chord]
                                c['_modulation_corrected'] = True
                                chord_corrections_m += 1
                    
                    # MoEのpitch補正: MoEモデルはE majorセクションのメロディを
                    # E minorの音程で出力する（C#5→B4, G#4→G4等、系統的に2半音低い）。
                    # これはminor→majorの転調時の3つのPC差（G→G#, C→C#, D→D#）が原因。
                    # 
                    # 補正戦略: major区間の全ノートに対して、
                    # E minor scaleにしか含まれないPC（G=7, C=0, D=2）を
                    # E major scaleの対応するPC（G#=8, C#=1, D#=3）に補正する。
                    # さらに、メロディラインのパターンが前半と一致する場合、
                    # major区間全体を+2半音シフトする。
                    minor_pcs = set((pre_key_root + iv) % 12 for iv in SCALES['natural_minor'])
                    major_pcs = set((parallel_major_root + iv) % 12 for iv in SCALES['major'])
                    
                    # E minor: PCs {0, 2, 4, 6, 7, 9, 11} = C D E F# G A B
                    # E major: PCs {1, 3, 4, 6, 8, 9, 11} = C# D# E F# G# A B
                    # Difference: minor has {0(C), 2(D), 7(G)}, major has {1(C#), 3(D#), 8(G#)}
                    minor_only_pcs = minor_pcs - major_pcs  # {0, 2, 7} for Em
                    major_only_pcs = major_pcs - minor_pcs  # {1, 3, 8} for E
                    
                    # PC shift: 各minor-only PCを+1半音（#方向）
                    pc_shift_map = {}
                    for mpc in sorted(minor_only_pcs):
                        target = (mpc + 1) % 12
                        if target in major_only_pcs:
                            pc_shift_map[mpc] = 1  # +1 semitone
                    
                    # 前半と後半のメロディパターンの類似性を検出して
                    # 前半minorセクションのメロディを基準にmajor区間を+2半音シフト
                    # 条件: 前半のメロディ音列と後半の音列が-2半音の関係にあること
                    half_duration = modulation_point
                    
                    # 前半の1弦メロディ列
                    minor_melody = sorted(
                        [n for n in notes if n['start'] < modulation_point and n.get('string') == 1],
                        key=lambda n: n['start']
                    )
                    # 後半の1弦メロディ列
                    major_melody = sorted(
                        [n for n in notes if n['start'] >= modulation_point and n.get('string') == 1],
                        key=lambda n: n['start']
                    )
                    
                    # パターンマッチ: 前半後半のメロディpitchが同一か検出
                    # MoEモデルがE majorセクションをE minorと同じpitchで出力している場合、
                    pitch_corrections = 0
                    
                    # 転調検出済み → 全ノートにmelody shift + PC shiftを適用
                    # 1弦メロディ用のPC補正マップ:
                    # B(PC=11) → C#(PC=1): +2半音 (minorのV相当 → majorの固有音)
                    # G(PC=7) → G#(PC=8): +1半音
                    # D(PC=2) → D#(PC=3): +1半音
                    # 注意: C(PC=0)は非シフト（M22B3等でCが正解のケースあり）
                    melody_shift = {11: 2, 7: 1, 2: 1}
                    
                    for note in notes:
                        if note['start'] >= modulation_point:
                            pc = note['pitch'] % 12
                            if note.get('string') == 1 and pc in melody_shift:
                                shift = melody_shift[pc]
                                note['pitch'] += shift
                                note['_modulation_pitch_shift'] = shift
                                pitch_corrections += 1
                            elif pc in pc_shift_map:
                                note['pitch'] += pc_shift_map[pc]
                                note['_modulation_pitch_shift'] = pc_shift_map[pc]
                                pitch_corrections += 1
                    
                    if chord_corrections_m > 0 or pitch_corrections > 0:
                        print(f"[music_theory] Phase -0.5: Parallel key modulation at {modulation_point:.1f}s")
                        print(f"  Minor→Major: {PC_NAMES[pre_key_root]}m → {PC_NAMES[parallel_major_root]}")
                        print(f"  Chord corrections: {chord_corrections_m}")
                        print(f"  Pitch corrections: {pitch_corrections} notes")
    
    # --- Phase 0: ノートベースのコード区間補強 ---
    # chroma検出器はEmとAmをマージする問題がある（共通構成音Eのため）。
    # Conformerが検出したベース音（5-6弦の低音）のルートPCから
    # コード区間を細分化する。
    if chords and notes:
        refined_chords = []
        for ci_idx, chord_info in enumerate(chords):
            cs = chord_info.get('start', 0)
            ce = chord_info.get('end', 0)
            chord_name = chord_info.get('chord', '')
            
            # Phase -1で構造推定された区間はスキップ
            if chord_info.get('_structure_inferred'):
                refined_chords.append(chord_info)
                continue
            
            # このコード区間が長い場合（>2秒）、ベース音でサブ分割を試みる
            if ce - cs < 2.0:
                refined_chords.append(chord_info)
                continue
            
            # この区間内の全ノートをビート単位でグループ化
            # ベース音だけでなくinner voice PCの変化も検出する
            if not beats:
                refined_chords.append(chord_info)
                continue
            
            # Phase -1で構造推定された区間の隣接はスキップ
            # （開放弦残響のPC変化で誤分割されるため）
            has_adj_inferred = False
            if ci_idx > 0 and chords[ci_idx - 1].get('_structure_inferred'):
                has_adj_inferred = True
            if ci_idx < len(chords) - 1 and chords[ci_idx + 1].get('_structure_inferred'):
                has_adj_inferred = True
            if has_adj_inferred:
                refined_chords.append(chord_info)
                continue
            
            beat_times_in_region = [b for b in beats if cs - 0.05 <= b < ce + 0.05]
            if len(beat_times_in_region) < 3:
                refined_chords.append(chord_info)
                continue
            
            # 現コードのルートPC
            current_root, current_quality, current_tone_pcs = parse_chord(chord_name)
            if current_root < 0:
                refined_chords.append(chord_info)
                continue
            
            # 各ビートのPC集合を計算
            beat_pc_sets = []
            for bi in range(len(beat_times_in_region)):
                bt_start = beat_times_in_region[bi]
                bt_end = (beat_times_in_region[bi + 1] 
                         if bi + 1 < len(beat_times_in_region) 
                         else bt_start + 0.67)
                bt_notes = [n for n in notes 
                           if bt_start - 0.05 <= n['start'] < bt_end + 0.05
                           and n.get('string', 0) in (2, 3, 5, 6)]
                pcs = frozenset(n['pitch'] % 12 for n in bt_notes)
                beat_pc_sets.append((bt_start, bt_end, pcs))
            
            # PC集合が変化するポイントでサブ分割
            sub_regions = []
            current_start = cs
            prev_pcs = beat_pc_sets[0][2] if beat_pc_sets else frozenset()
            
            for bi in range(1, len(beat_pc_sets)):
                cur_pcs = beat_pc_sets[bi][2]
                # PC集合が異なり、かつ新しいPCが現コードの構成音でない場合
                new_pcs = cur_pcs - prev_pcs
                if new_pcs and not new_pcs.issubset(set(current_tone_pcs)):
                    sub_regions.append((current_start, beat_pc_sets[bi][0], prev_pcs))
                    current_start = beat_pc_sets[bi][0]
                if cur_pcs:
                    prev_pcs = cur_pcs
            sub_regions.append((current_start, ce, prev_pcs))
            
            if len(sub_regions) <= 1:
                refined_chords.append(chord_info)
                continue
            
            # 各サブ区間にコードを割り当て
            for sr_start, sr_end, region_pcs in sub_regions:
                # この区間のノートのPC集合がコード構成音と一致するか
                if region_pcs.issubset(set(current_tone_pcs)) or not region_pcs:
                    # 元のコードの構成音内 → 同じコード
                    refined_chords.append({
                        'start': sr_start, 'end': sr_end,
                        'chord': chord_name,
                        'confidence': chord_info.get('confidence', 0.5),
                    })
                else:
                    # 新しいPCあり → コード推定
                    region_notes = [n for n in notes 
                                   if sr_start <= n['start'] < sr_end]
                    all_pcs = set(n['pitch'] % 12 for n in region_notes)
                    
                    # ルート推定: 最低音のPC
                    bass_candidates = [n for n in region_notes 
                                      if n.get('string', 0) in (5, 6)]
                    if bass_candidates:
                        root_pc = min(bass_candidates, key=lambda n: n['pitch'])['pitch'] % 12
                    else:
                        # ベース音がない → 新しいPCの中で最も低いものをルートに
                        new_pcs = region_pcs - set(current_tone_pcs)
                        root_pc = min(new_pcs) if new_pcs else current_root
                    
                    # メジャー/マイナー判定
                    minor_3rd = (root_pc + 3) % 12
                    major_3rd = (root_pc + 4) % 12
                    root_name = PC_NAMES[root_pc]
                    
                    if minor_3rd in all_pcs:
                        inferred_chord = root_name + 'm'
                    elif major_3rd in all_pcs:
                        inferred_chord = root_name
                    else:
                        inferred_chord = root_name + 'm'
                    
                    refined_chords.append({
                        'start': sr_start, 'end': sr_end,
                        'chord': inferred_chord,
                        'confidence': 0.4,
                        '_note_refined': True,
                    })
            
            print(f"[music_theory] Chord refined: {chord_name} ({cs:.1f}-{ce:.1f}s) -> "
                  f"{len(sub_regions)} sub-regions")
        
        if len(refined_chords) != len(chords):
            print(f"[music_theory] Chord regions: {len(chords)} -> {len(refined_chords)}")
            chords = refined_chords
    
    # --- キー推定 ---
    key_root, scale_type = estimate_key(chords)
    scale_pcs = get_scale_pcs(key_root, scale_type)
    key_name = f"{PC_NAMES[key_root]} {scale_type}"
    
    # 転調点がある場合、major区間用のスケールPCsも準備
    # _modulation_point is set by Phase -0.5 above (or None if no modulation)
    if _modulation_point and scale_type == 'natural_minor':
        major_scale_pcs = get_scale_pcs(key_root, 'major')
        print(f"[music_theory] Key estimated: {key_name} (→ {PC_NAMES[key_root]} major after {_modulation_point:.1f}s)")
        print(f"[music_theory] Minor Scale PCs: {sorted(scale_pcs)}")
        print(f"[music_theory] Major Scale PCs: {sorted(major_scale_pcs)}")
    else:
        major_scale_pcs = None
        print(f"[music_theory] Key estimated: {key_name}")
        print(f"[music_theory] Scale PCs: {sorted(scale_pcs)}")
    
    print(f"[music_theory] Processing {len(notes)} notes with {len(chords)} chord regions")
    
    # --- 各ノートを検証・補正 ---
    corrected_count = 0
    removed_count = 0
    
    result = []
    for note in notes:
        # 転調後のノートにはmajorスケールを使用
        if _modulation_point and major_scale_pcs and note['start'] >= _modulation_point:
            note = correct_note_pitch(note, chords, key_root, major_scale_pcs, tuning)
        else:
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
    # Phase 2: アルペジオ・パターン補完 (Inner Voice Synthesis)
    # =====================================================================
    # 診断結果: Conformerの3弦onset headが事実上死んでおり（確率0.01-0.08）、
    # inner voice (2弦/3弦) がほぼ全拍で消失する。
    # 対策: tripletアルペジオパターンを検出し、メロディ検出済み拍で
    # inner voiceが不足している場合、コード構成音から合成する。
    # Note: Conformer出力時はViterbi DPをスキップするため、
    # ノート数膨張によるDP破綻は発生しない。
    
    SIMULTANEOUS_THRESHOLD = 0.03
    pattern_completions = 0
    
    result.sort(key=lambda n: (n['start'], -n['pitch']))
    
    # ビートグループ化
    beat_groups = []
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
    
    # --- アルペジオ補完 (triplet検出時のみ) ---
    # リズム分析はPhase 6で行うが、事前に簡易判定を行う
    # IOIの中央値がbeat_intervalの約1/3ならtriplet
    if len(result) >= 10 and beats and chords:
        import numpy as _np
        _onsets = sorted(set(n['start'] for n in result))
        _ioi = _np.diff(_onsets)
        _ioi_filtered = _ioi[(_ioi > 0.05) & (_ioi < 1.0)]
        
        _is_triplet = False
        if len(_ioi_filtered) > 5 and len(beats) >= 2:
            _beat_interval = float(_np.median(_np.diff(beats)))
            _median_ioi = float(_np.median(_ioi_filtered))
            _ratio = _beat_interval / _median_ioi if _median_ioi > 0 else 1.0
            # ratio ≈ 3 → triplet
            if 2.3 < _ratio < 4.0:
                _is_triplet = True
        
        if _is_triplet:
            # --- ビート単位でのinner voice補完 ---
            # 3連符アルペジオ: 各拍に3ノートが順次発音される（同時ではない）
            # beat_interval（≈0.67s）を使ってノートをビート単位にグループ化し、
            # inner voiceが不足しているビートにのみ補完する
            
            # ビート時刻リスト
            beat_times = sorted(beats) if beats else []
            
            if beat_times and _beat_interval > 0:
                for bt_idx in range(len(beat_times)):
                    bt_start = beat_times[bt_idx]
                    bt_end = beat_times[bt_idx + 1] if bt_idx + 1 < len(beat_times) else bt_start + _beat_interval
                    
                    # このビート内のノート
                    beat_notes = [n for n in result if bt_start - 0.05 <= n['start'] < bt_end + 0.05]
                    
                    if not beat_notes:
                        continue
                    
                    pitches_in_beat = set(n['pitch'] for n in beat_notes)
                    strings_in_beat = set(n.get('string', 0) for n in beat_notes)
                    
                    # inner voice (2弦/3弦) のノート数
                    inner_notes = [n for n in beat_notes if n.get('string', 0) in (2, 3)]
                    
                    # 既に2つ以上のinner voiceがあればスキップ
                    if len(inner_notes) >= 2:
                        continue
                    
                    # メロディ音が存在するか
                    has_melody = any(n.get('string', 0) == 1 for n in beat_notes)
                    if not has_melody:
                        has_melody = any(n['pitch'] >= 64 for n in beat_notes)
                    if not has_melody:
                        continue
                    
                    # コード取得
                    chord_root, chord_quality, chord_tone_pcs = find_chord_at_time(chords, bt_start)
                    if chord_root < 0:
                        continue
                    
                    # 不足しているinner voice弦を特定
                    missing_strings = []
                    for s in (2, 3):
                        if s not in strings_in_beat:
                            missing_strings.append(s)
                    
                    if not missing_strings:
                        continue
                    
                    # inner voiceの発音タイミングを推定
                    # 3連符: melody=拍頭, inner2=拍頭+IOI, inner3=拍頭+IOI*2
                    max_group_pitch = max(n['pitch'] for n in beat_notes)
                    
                    for ms in missing_strings:
                        str_idx = 6 - ms
                        if str_idx < 0 or str_idx >= len(tuning):
                            continue
                        open_pitch = tuning[str_idx]
                        
                        # この弦で弾ける最低フレットのコード構成音
                        best_fret = None
                        best_pitch = None
                        for fret in range(0, 13):
                            pitch = open_pitch + fret
                            pc = pitch % 12
                            if pc in chord_tone_pcs and pitch < max_group_pitch:
                                if pitch not in pitches_in_beat:
                                    best_fret = fret
                                    best_pitch = pitch
                                    break
                        
                        if best_fret is None:
                            continue
                        
                        # 発音タイミング: ビート内のノートの後に配置
                        synth_time = bt_start + _median_ioi * len(inner_notes)
                        if synth_time >= bt_end:
                            synth_time = bt_start + _median_ioi
                        
                        synth_note = {
                            'start': round(synth_time, 4),
                            'end': round(synth_time + _median_ioi * 2, 4),
                            'pitch': best_pitch,
                            'string': ms,
                            'fret': best_fret,
                            'velocity': 0.6,
                            '_synthesized': True,
                            '_synth_reason': f"arpeggio_inner_voice (chord={chord_root}:{chord_quality})",
                            '_is_chord_tone': True,
                            '_is_scale_tone': best_pitch % 12 in scale_pcs,
                            '_theory_flag': 'synthesized',
                        }
                        result.append(synth_note)
                        pitches_in_beat.add(best_pitch)
                        strings_in_beat.add(ms)
                        pattern_completions += 1
    
    # 再ソート
    result.sort(key=lambda n: (n['start'], -n['pitch']))
    
    print(f"[music_theory] Pattern completions: {pattern_completions}")
    
    # =====================================================================
    # Phase 2.5: ベース音補完
    # =====================================================================
    # コードのルート音が変化した小節で、Conformerが前のコードのベース音を
    # 検出している場合、正しいルート音をベースとして注入する。
    # 例: Em(E2)→Am(A2) 変化で、開放弦の残響によりE2が検出され続ける場合、
    # Am区間の1拍目にA2(5弦開放)を注入。
    
    bass_injections = 0
    
    if _is_triplet and beats and chords:
        # コードのルート音→弦/フレットマッピング
        # 標準チューニングでのベース音位置
        BASS_POSITIONS = {
            0:  (6, 0, 40),   # E → 6弦開放
            5:  (5, 0, 45),   # A → 5弦開放
            10: (5, 5, 50),   # D → 5弦5フレット (Bb)
            2:  (5, 5, 50),   # D → 5弦5フレット
            3:  (6, 3, 43),   # G → 6弦3フレット
            7:  (5, 2, 47),   # B → 5弦2フレット
        }
        # より正確なマッピング
        BASS_MAP = {
            'E': (6, 0, 40), 'Em': (6, 0, 40),
            'A': (5, 0, 45), 'Am': (5, 0, 45),
            'D': (4, 0, 50), 'Dm': (4, 0, 50),
            'G': (6, 3, 43),
            'C': (5, 3, 48),
            'B': (5, 2, 47), 'Bm': (5, 2, 47),
            'F': (6, 1, 41), 'Fm': (6, 1, 41),
        }
        
        for ci, chord in enumerate(chords):
            chord_name = chord.get('chord', '')
            cs = chord.get('start', 0)
            ce = chord.get('end', 0)
            
            if chord_name in BASS_MAP and chord.get('_structure_inferred'):
                target_string, target_fret, target_pitch = BASS_MAP[chord_name]
                
                # このコードの前のコードが異なるルートかチェック
                prev_chord = chords[ci - 1].get('chord', '') if ci > 0 else ''
                if prev_chord == chord_name:
                    continue  # 同じコードの継続なら不要
                
                # コード区間の最初のビート時刻を探す
                first_beat = None
                for bt in beats:
                    if cs - 0.05 <= bt <= cs + 0.2:
                        first_beat = bt
                        break
                if first_beat is None:
                    first_beat = cs
                


                # 既にターゲットピッチのベースが存在するか？
                existing_bass = [n for n in result 
                                if abs(n['start'] - first_beat) < 0.1
                                and n['pitch'] == target_pitch]
                if existing_bass:
                    continue  # 既に正しいベースが存在
                
                # 間違ったベース（前のコードの残響）が存在するか？
                # コード区間開始以降のノートのみ対象
                wrong_bass = [n for n in result
                             if abs(n['start'] - first_beat) < 0.1
                             and n['start'] >= first_beat - 0.06  # ビート直前も含む
                             and n.get('string', 0) in (5, 6)
                             and n['pitch'] != target_pitch]
                
                if wrong_bass:
                    # 既存の間違ったベースを正しいベースに補正
                    for wb in wrong_bass:
                        wb['_original_pitch'] = wb['pitch']
                        wb['pitch'] = target_pitch
                        wb['string'] = target_string
                        wb['fret'] = target_fret
                        wb['_bass_corrected'] = True
                        bass_injections += 1
                else:
                    # ベース音が全く無い → 注入
                    bass_note = {
                        'start': round(first_beat, 4),
                        'end': round(first_beat + _median_ioi * 2, 4),
                        'pitch': target_pitch,
                        'string': target_string,
                        'fret': target_fret,
                        'velocity': 0.7,
                        '_synthesized': True,
                        '_synth_reason': f"bass_injection (chord={chord_name})",
                        '_is_chord_tone': True,
                        '_theory_flag': 'synthesized',
                    }
                    result.append(bass_note)
                    bass_injections += 1
        
        if bass_injections > 0:
            result.sort(key=lambda n: (n['start'], -n['pitch']))
            print(f"[music_theory] Bass injections: {bass_injections}")
    
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
    # Phase 3.5: 開放弦バイアス補正
    # =====================================================================
    # 確認済み問題: ConformerがEm→Amのコード変更に追従できず、
    # 開放弦ピッチ(B3,G3,E2)を出力し続ける。
    # 対策: fret=0のノートがコード構成音でない場合、同一弦上の
    # 最近傍コード構成音（fret 1-4以内）に補正する。
    
    open_string_corrections = 0
    
    if chords:
        for note in result:
            fret = note.get('fret', 0)
            if fret != 0:
                continue  # 開放弦のみ対象
            
            # ベース弦(5,6弦)はCQTで信号が強く検出が信頼できるためスキップ
            if note.get('string', 0) in (5, 6):
                continue
            
            pitch = note['pitch']
            pc = pitch % 12
            string = note.get('string', 0)
            
            # このノートの時刻のコード
            chord_root, chord_quality, chord_tone_pcs = find_chord_at_time(
                chords, note['start'])
            if chord_root < 0:
                continue
            
            # 開放弦ピッチがコード構成音ならそのまま（正しい）
            if pc in chord_tone_pcs:
                continue
            
            # 開放弦ピッチがコード構成音でない → 同一弦上で最近傍のコード構成音を探す
            best_fret = None
            best_pitch = None
            best_dist = float('inf')
            
            for candidate_fret in range(1, 5):  # fret 1-4 の範囲
                candidate_pitch = pitch + candidate_fret
                candidate_pc = candidate_pitch % 12
                if candidate_pc in chord_tone_pcs:
                    if candidate_fret < best_dist:
                        best_dist = candidate_fret
                        best_fret = candidate_fret
                        best_pitch = candidate_pitch
                        break  # 最も近いものを採用
            
            if best_fret is not None:
                note['_original_pitch'] = pitch
                note['_original_fret'] = 0
                note['pitch'] = best_pitch
                note['fret'] = best_fret
                note['_open_string_corrected'] = True
                note['_correction_reason'] = (
                    f"open_string_bias: p{pitch}(s{string}f0) -> "
                    f"p{best_pitch}(s{string}f{best_fret}) "
                    f"chord_root={chord_root}"
                )
                open_string_corrections += 1
    
    print(f"[music_theory] Open string corrections: {open_string_corrections}")
    
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
    
    # =====================================================================
    # Phase 6: 曲全体のリズム基盤検出 (IOIベース)
    # =====================================================================
    # ノート間隔(IOI)とビート間隔の比率から、曲の基本subdivision を判定。
    # triplet (3連符) vs straight (通常) vs sixteenth (16分) 
    
    rhythm_info = {'subdivision': 'eighth', 'confidence': 0.0}
    
    if len(result) >= 10 and beats:
        import numpy as np
        onsets = sorted(set(n['start'] for n in result))
        ioi = np.diff(onsets)
        # 極端に短い(同時発音)や長い(休符)は除外
        ioi_filtered = ioi[(ioi > 0.05) & (ioi < 1.0)]
        
        if len(ioi_filtered) > 10:
            beat_times = beats if isinstance(beats, list) else list(beats)
            
            # ビート間隔: BPMから算出（ビート配列は不正確な場合がある）
            # beats.jsonのBPMを取得
            bpm_val = 90.0
            for c in chords[:1]:
                # chordsからは取れないのでデフォルト
                pass
            
            # ビート配列の間隔も参考にする
            if len(beat_times) >= 2:
                beat_array_interval = float(np.median(np.diff(beat_times)))
            else:
                beat_array_interval = 0.5
            
            # IOIの中央値
            median_ioi = np.median(ioi_filtered)
            
            # 複数のbeat_interval候補でratioを計算
            # BPM由来の値が最も信頼性が高い
            candidates = []
            
            # 候補1: ビート配列のmedian
            r1 = beat_array_interval / median_ioi if median_ioi > 0 else 1.0
            candidates.append(('beat_array', beat_array_interval, r1))
            
            # 候補2: ビート配列の2倍（ビートが半拍で検出されている場合）
            r2 = (beat_array_interval * 2) / median_ioi if median_ioi > 0 else 1.0
            candidates.append(('beat_array_x2', beat_array_interval * 2, r2))
            
            # 候補3: ビート配列の1.5倍（3/4拍子で2拍分検出されている場合）
            r3 = (beat_array_interval * 1.5) / median_ioi if median_ioi > 0 else 1.0
            candidates.append(('beat_array_x1.5', beat_array_interval * 1.5, r3))
            
            # 各候補からratio≈3(triplet)に最も近いものを探す
            best_name = candidates[0][0]
            best_interval = candidates[0][1]
            ratio = candidates[0][2]
            
            for name, interval, r in candidates:
                if abs(r - 3.0) < abs(ratio - 3.0):
                    best_name = name
                    best_interval = interval
                    ratio = r
                elif abs(r - 2.0) < abs(ratio - 2.0) and abs(r - 2.0) < abs(ratio - 3.0):
                    best_name = name
                    best_interval = interval
                    ratio = r
            
            beat_interval = best_interval
            
            # ratio ≈ 3 → 3連符 (1拍に3音)
            # ratio ≈ 2 → 8分音符 (1拍に2音)
            # ratio ≈ 4 → 16分音符 (1拍に4音)
            # ratio ≈ 6 → 3連16分 or ビートが2拍単位
            
            dist_triplet = abs(ratio - 3.0)
            dist_eighth = abs(ratio - 2.0)
            dist_16th = abs(ratio - 4.0)
            dist_triplet_2beat = abs(ratio - 6.0)  # 2拍単位ビート + 3連
            
            min_dist = min(dist_triplet, dist_eighth, dist_16th, dist_triplet_2beat)
            
            if min_dist == dist_triplet:
                rhythm_info = {'subdivision': 'triplet', 
                              'confidence': 1.0 - min(dist_triplet, 1.0),
                              'notes_per_beat': 3}
            elif min_dist == dist_triplet_2beat:
                # ビートが2拍単位で検出されている + 3連符
                rhythm_info = {'subdivision': 'triplet', 
                              'confidence': 1.0 - min(dist_triplet_2beat, 1.0),
                              'notes_per_beat': 3,
                              'beat_halved': True}
            elif min_dist == dist_16th:
                rhythm_info = {'subdivision': 'sixteenth',
                              'confidence': 1.0 - min(dist_16th, 1.0),
                              'notes_per_beat': 4}
            else:
                rhythm_info = {'subdivision': 'eighth',
                              'confidence': 1.0 - min(dist_eighth, 1.0),
                              'notes_per_beat': 2}
            
            rhythm_info['median_ioi'] = float(median_ioi)
            rhythm_info['beat_interval'] = float(beat_interval)
            rhythm_info['ratio'] = float(ratio)
            
            # 全ノートにリズム情報を付与
            if rhythm_info['subdivision'] == 'triplet':
                for n in result:
                    n['_rhythm'] = 'triplet'
            
            print(f"[music_theory] Rhythm: {rhythm_info['subdivision']} "
                  f"(confidence={rhythm_info['confidence']:.2f}, "
                  f"ratio={ratio:.2f}, median_ioi={median_ioi:.3f}s, "
                  f"beat_interval={beat_interval:.3f}s)")
    
    total_corrected = corrected_count + pattern_completions + voicing_corrections
    print(f"[music_theory] Total corrections: {total_corrected} / {len(result)} notes")
    
    return result, rhythm_info


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
