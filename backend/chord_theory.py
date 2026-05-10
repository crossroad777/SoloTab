"""
chord_theory.py — 音楽理論エンジン (コード解析・典型フォームDB)
================================================================
坂井論文 2024「主旋律と和音を同時に演奏するソロギターのためのタブ譜生成」準拠。

機能:
  - コード名解析 (_parse_chord_name)
  - コード構成音取得 (_get_chord_notes_pc)
  - 典型フォームDB照合 (_typical_form_match_cost)
  - 音楽理論出力コスト (_music_theory_output_cost)
  - コードフォーム内ポジション優先 (_chord_form_position_cost)
"""

from typing import List, Tuple, Optional
import json
import os


# =============================================================================
# 典型フォームDB
# =============================================================================

_CHORD_FORMS_DB = None
_CHORD_FORMS_LOOKUP = None

def _load_chord_forms_db():
    """典型フォームDBをロードする（遅延初期化）。"""
    global _CHORD_FORMS_DB, _CHORD_FORMS_LOOKUP
    if _CHORD_FORMS_DB is not None:
        return
    db_path = os.path.join(os.path.dirname(__file__), "chord_forms_db.json")
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            _CHORD_FORMS_DB = json.load(f)
        # ルックアップテーブル構築
        _CHORD_FORMS_LOOKUP = {}
        for form in _CHORD_FORMS_DB:
            chord = form["chord"]
            if chord not in _CHORD_FORMS_LOOKUP:
                _CHORD_FORMS_LOOKUP[chord] = []
            _CHORD_FORMS_LOOKUP[chord].append(form)
    else:
        _CHORD_FORMS_DB = []
        _CHORD_FORMS_LOOKUP = {}


# =============================================================================
# コード構成音マッピング
# =============================================================================

# ルートからの半音数
_CHORD_INTERVALS = {
    "major": [0, 4, 7], "minor": [0, 3, 7], "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11], "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9], "aug": [0, 4, 8], "sus4": [0, 5, 7],
    "sus2": [0, 2, 7], "m7b5": [0, 3, 6, 10], "6": [0, 4, 7, 9],
}

# コード名 → DB検索キー変換用の定数
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_QUALITY_SUFFIX = {
    "major": "", "minor": "m", "7": "7", "m7": "m7", "maj7": "maj7",
    "dim": "dim", "dim7": "dim7", "aug": "aug", "sus4": "sus4", "sus2": "sus2",
    "m7b5": "m7b5", "6": "6",
}


def _parse_chord_name(chord_str: str) -> Tuple[int, str]:
    """コード名からルートPCとクオリティを抽出。
    例: 'Am7' -> (9, 'm7'), 'C#dim' -> (1, 'dim'), 'G' -> (7, 'major')
    """
    if not chord_str or chord_str in ('N', 'X', 'NC', 'N.C.'):
        return (-1, '')
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    s = chord_str.strip()
    if not s or s[0] not in note_map:
        return (-1, '')
    root_pc = note_map[s[0]]
    idx = 1
    if idx < len(s) and s[idx] == '#':
        root_pc = (root_pc + 1) % 12
        idx += 1
    elif idx < len(s) and s[idx] == 'b':
        root_pc = (root_pc - 1) % 12
        idx += 1
    quality_str = s[idx:].lower()
    if quality_str in ('', 'maj'):
        quality = 'major'
    elif quality_str in ('m', 'min', 'mi'):
        quality = 'minor'
    elif quality_str in ('7', 'dom7'):
        quality = '7'
    elif quality_str in ('m7', 'min7', 'mi7'):
        quality = 'm7'
    elif quality_str in ('maj7', 'ma7'):
        quality = 'maj7'
    elif quality_str in ('dim', 'o'):
        quality = 'dim'
    elif quality_str in ('dim7', 'o7'):
        quality = 'dim7'
    elif quality_str in ('aug', '+'):
        quality = 'aug'
    elif quality_str in ('sus4', 'sus'):
        quality = 'sus4'
    elif quality_str == 'sus2':
        quality = 'sus2'
    elif quality_str in ('m7b5', 'hdim', 'hdim7'):
        quality = 'm7b5'
    else:
        quality = 'major'  # デフォルト
    return (root_pc, quality)


def _get_chord_notes_pc(root_pc: int, quality: str) -> List[int]:
    """コードの構成音のピッチクラスリストを返す。"""
    intervals = _CHORD_INTERVALS.get(quality, [0, 4, 7])
    return [(root_pc + iv) % 12 for iv in intervals]


def _get_chord_at_time(chords: List[dict], time_sec: float) -> Optional[dict]:
    """指定時刻のコード情報を返す。"""
    if not chords:
        return None
    for chord in chords:
        start = chord.get('start', chord.get('time', 0))
        end = chord.get('end', start + chord.get('duration', 999))
        if start <= time_sec < end:
            return chord
    return None


def _typical_form_match_cost(combo: Tuple[Tuple[int, int], ...],
                             chord_name: str,
                             tuning: List[int]) -> float:
    """
    典型フォーム一致コスト (坂井論文 3.6 typical(qn) に相当)。

    combo の押弦パターンが典型フォームDBのいずれかと一致すれば低コスト、
    一致しなければ高コスト。

    Returns
    -------
    float : 一致すれば 0.0, 部分一致なら 5.0, 不一致なら 12.0
    """
    _load_chord_forms_db()
    if not _CHORD_FORMS_LOOKUP or not chord_name:
        return 0.0  # DB未ロードまたはコード不明時はペナルティなし

    # comboから6弦のフレット配列を構築
    combo_frets = [-1] * 6  # 6弦→1弦
    for s, f in combo:
        idx = 6 - s  # string 6→index 0, string 1→index 5
        if 0 <= idx < 6:
            combo_frets[idx] = f

    # コード名の正規化（DBのキーと照合）
    root_pc, quality = _parse_chord_name(chord_name)
    if root_pc < 0:
        return 0.0

    db_key = _NOTE_NAMES[root_pc] + _QUALITY_SUFFIX.get(quality, "")

    forms = _CHORD_FORMS_LOOKUP.get(db_key, [])
    if not forms:
        return 0.0  # DBにコードがない場合はペナルティなし

    best_match = 0  # 一致した弦の数
    for form in forms:
        db_frets = form["frets"]
        match_count = 0
        for i in range(6):
            if combo_frets[i] >= 0 and db_frets[i] >= 0:
                if combo_frets[i] == db_frets[i]:
                    match_count += 1
        best_match = max(best_match, match_count)

    sounding = sum(1 for f in combo_frets if f >= 0)
    if sounding == 0:
        return 0.0

    match_ratio = best_match / sounding
    if match_ratio >= 0.8:
        return 0.0    # ほぼ完全一致
    elif match_ratio >= 0.5:
        return 5.0    # 部分一致
    else:
        return 12.0   # 不一致


def _music_theory_output_cost(combo: Tuple[Tuple[int, int], ...],
                              chord_name: str,
                              tuning: List[int]) -> float:
    """
    音楽理論に基づく出力コスト (坂井論文 3.6 C((xn,cn)|qn) に相当)。

    1. 同時発音数コスト voices(qn): 発音数が多いほど響きが豊か → 報酬
    2. 典型フォーム一致コスト typical(qn)
    3. ルート音制約: 最低音がコードのルート音と一致 → 報酬
    4. 構成音一致: 発音する音がコード構成音に含まれている → 報酬
    """
    cost = 0.0

    if not chord_name:
        return 0.0

    root_pc, quality = _parse_chord_name(chord_name)
    if root_pc < 0:
        return 0.0

    chord_pcs = _get_chord_notes_pc(root_pc, quality)

    # --- 1. 同時発音数コスト (voices) ---
    n_sounding = len(combo)
    if n_sounding >= 2:
        cost += -3.0 * n_sounding  # 発音数が多いほど報酬

    # --- 2. 典型フォーム一致コスト ---
    cost += _typical_form_match_cost(combo, chord_name, tuning)

    # --- 3. ルート音 = 最低音 制約 ---
    if n_sounding >= 2:
        # 各弦のピッチを計算
        pitches = []
        for s, f in combo:
            idx = 6 - s
            if 0 <= idx < len(tuning):
                pitches.append((tuning[idx] + f, s))
        if pitches:
            pitches.sort()  # 低い音から
            bass_pc = pitches[0][0] % 12
            if bass_pc == root_pc:
                cost += -15.0  # ルート音がベースにある → 大きな報酬
            else:
                cost += 8.0    # ルート音がベースにない → ペナルティ

    # --- 4. 構成音一致 ---
    for s, f in combo:
        idx = 6 - s
        if 0 <= idx < len(tuning):
            pc = (tuning[idx] + f) % 12
            if pc in chord_pcs:
                cost += -2.0  # 構成音一致 → 報酬
            else:
                cost += 3.0   # 非構成音 → ペナルティ

    return cost


def _chord_form_position_cost(s: int, f: int, chord_name: str,
                               tuning: List[int]) -> float:
    """
    ソロギター用: 単音がコードフォーム内にあるかのコスト。

    改良版: 複数フォームにマッチする場合、以下の優先順位で選択:
    1. オープンフォーム > バレーフォーム
    2. 低ポジション > 高ポジション
    3. フォーム中心からの距離が近いほど大きなボーナス
    """
    _load_chord_forms_db()
    if not _CHORD_FORMS_LOOKUP or not chord_name:
        return 0.0

    root_pc, quality = _parse_chord_name(chord_name)
    if root_pc < 0:
        return 0.0

    db_key = _NOTE_NAMES[root_pc] + _QUALITY_SUFFIX.get(quality, "")
    forms = _CHORD_FORMS_LOOKUP.get(db_key, [])
    if not forms:
        return 0.0

    idx = 6 - s  # string番号→配列index
    if idx < 0 or idx >= 6:
        return 0.0

    # マッチするフォームを全て収集し、ベストを選択
    best_bonus = 0.0
    for form in forms:
        db_frets = form["frets"]
        if db_frets[idx] != f:
            continue

        form_pos = form.get("position", 0)
        source = form.get("source", "")

        # フォームの「中心フレット」を計算（押弦フレットの平均）
        pressed = [fr for fr in db_frets if fr > 0]
        form_center = sum(pressed) / len(pressed) if pressed else 0

        # ベースボーナス: フォームの種別で決定
        if source in ("open", "extra") and form_pos == 0:
            base_bonus = -12.0  # オープンフォームは最優先
        elif form_pos <= 2:
            base_bonus = -6.0   # ローポジション
        elif form_pos <= 7:
            base_bonus = -3.0   # 中ポジション
        else:
            base_bonus = -1.0   # ハイポジション

        # 距離減衰: ノートのフレットがフォーム中心から離れるほどボーナス減少
        dist = abs(f - form_center)
        decay = max(0.3, 1.0 - dist * 0.15)
        bonus = base_bonus * decay

        best_bonus = min(best_bonus, bonus)

    if best_bonus < 0:
        return best_bonus

    # コードの構成音かどうかもチェック
    chord_pcs = _get_chord_notes_pc(root_pc, quality)
    if 0 <= (6 - s) < len(tuning):
        pc = (tuning[6 - s] + f) % 12
        if pc in chord_pcs:
            return -3.0  # コード構成音だがフォーム外 → 小さいボーナス

    return 0.0  # フォーム外・非構成音 → ペナルティなし
