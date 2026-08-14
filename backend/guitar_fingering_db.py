"""
guitar_fingering_db.py — コードフォーム・スケールパターンDB v3
============================================================
データソース（優先順）:
  1. chords-db (tombatossals): 3,283ボイシング（人間キュレーション済み）
  2. chord-collection (T-vK): 167,011ボイシング（自動生成、フォールバック）
  3. 手動追加: Segovia式スケールパターン
  4. 逆算ルール: 3,283ボイシングからのデータ駆動確率テーブル
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json, os

_DIR = os.path.dirname(__file__)

# --- 2層コードDB ---
_CHORDS_DB_PATH = os.path.join(_DIR, '..', 'datasets',
                                'chords-db', 'lib', 'guitar.json')
_CHORD_COLLECTION_PATH = os.path.join(_DIR, '..', 'datasets',
                                       'chord-collection', 'chords.complete.json')
_CURATED_DB = None   # 3,283 (優先)
_MASS_DB = None       # 167,011 (フォールバック)


def _load_curated_db():
    """chords-db (tombatossals) — 人間キュレーション済み、高品質"""
    global _CURATED_DB
    if _CURATED_DB is not None:
        return _CURATED_DB
    try:
        with open(_CHORDS_DB_PATH, 'r') as f:
            raw = json.load(f)
        _CURATED_DB = {}
        for key_chords in raw.get('chords', {}).values():
            for chord in key_chords:
                chord_name = f"{chord['key']}_{chord['suffix']}"
                positions = []
                for pos in chord.get('positions', []):
                    positions.append({
                        'frets': pos['frets'],
                        'fingers': pos['fingers'],
                        'baseFret': pos['baseFret'],
                        'barres': pos.get('barres', []),
                    })
                _CURATED_DB[chord_name] = positions
        n_voicings = sum(len(v) for v in _CURATED_DB.values())
        print(f"[fingering_db] curated DB: {len(_CURATED_DB)} chords, {n_voicings} voicings")
    except Exception as e:
        print(f"[fingering_db] curated DB load failed: {e}")
        _CURATED_DB = {}
    return _CURATED_DB


def _load_mass_db():
    """chord-collection (T-vK) — 167K、自動生成、フォールバック用"""
    global _MASS_DB
    if _MASS_DB is not None:
        return _MASS_DB
    try:
        with open(_CHORD_COLLECTION_PATH, 'r') as f:
            raw = json.load(f)
        _MASS_DB = {}
        for chord_name, voicing_list in raw.items():
            positions = []
            for v in voicing_list:
                pos_raw = v.get('positions', [])
                fing_raw = v.get('fingerings', [[]])
                if len(pos_raw) != 6:
                    continue
                # フレットをint変換
                frets = []
                for p in pos_raw:
                    if p == 'x':
                        frets.append(-1)
                    else:
                        frets.append(int(p))
                # 指番号（最初のfingering候補を使用）
                fingers = [0] * 6
                if fing_raw and len(fing_raw[0]) == 6:
                    fingers = [int(x) for x in fing_raw[0]]
                # baseFretを推定（最小の非0フレット）
                fretted = [f for f in frets if f > 0]
                base_fret = min(fretted) if fretted else 1
                positions.append({
                    'frets_abs': frets,      # 絶対フレット番号
                    'fingers': fingers,
                    'baseFret': base_fret,
                })
            if positions:
                _MASS_DB[chord_name] = positions
        n_voicings = sum(len(v) for v in _MASS_DB.values())
        print(f"[fingering_db] mass DB: {len(_MASS_DB)} chords, {n_voicings} voicings")
    except Exception as e:
        print(f"[fingering_db] mass DB load failed: {e}")
        _MASS_DB = {}
    return _MASS_DB


# --- 逆算ルール: データ駆動の確率テーブル ---
OFFSET_TO_FINGER_PROB = {
    0: {1: 0.96, 2: 0.03, 3: 0.01, 4: 0.00},
    1: {2: 0.55, 3: 0.23, 1: 0.16, 4: 0.07},
    2: {3: 0.50, 4: 0.31, 2: 0.15, 1: 0.04},
    3: {4: 0.78, 3: 0.15, 2: 0.06, 1: 0.01},
}

BARRE_FINGER_PROB = {1: 0.88, 2: 0.06, 4: 0.04, 3: 0.03}

STRING_FINGER_PROB = {
    6: {1: 0.53, 2: 0.27, 3: 0.11, 4: 0.09},
    5: {1: 0.44, 2: 0.26, 3: 0.21, 4: 0.09},
    4: {1: 0.45, 3: 0.24, 2: 0.20, 4: 0.10},
    3: {1: 0.39, 3: 0.24, 2: 0.22, 4: 0.15},
    2: {1: 0.37, 4: 0.27, 3: 0.21, 2: 0.15},
    1: {1: 0.48, 4: 0.32, 3: 0.12, 2: 0.08},
}


def fret_to_finger_in_position(fret: int, position: int) -> int:
    """ポジション内のフレットから最も確率の高い指番号を返す。"""
    if fret == 0:
        return 0
    offset = fret - position
    if offset in OFFSET_TO_FINGER_PROB:
        probs = OFFSET_TO_FINGER_PROB[offset]
        return max(probs, key=probs.get)
    elif offset < 0:
        return 1
    elif offset > 3:
        return 4
    return 1


def _parse_chord_name(chord_name: str):
    """コード名をroot + suffixに分解"""
    root = ""
    suffix = ""
    for r in ['C#', 'Db', 'D#', 'Eb', 'F#', 'Gb', 'G#', 'Ab', 'A#', 'Bb',
              'C', 'D', 'E', 'F', 'G', 'A', 'B']:
        if chord_name.startswith(r):
            root = r
            suffix = chord_name[len(r):]
            break
    return root, suffix.strip()


_ENHARMONIC = {
    'C#': 'Db', 'Db': 'C#', 'D#': 'Eb', 'Eb': 'D#',
    'F#': 'Gb', 'Gb': 'F#', 'G#': 'Ab', 'Ab': 'G#',
    'A#': 'Bb', 'Bb': 'A#',
}


def _match_curated(note_frets: dict, voicings: list) -> dict:
    """curated DB (chords-db形式) からベストマッチを検索"""
    best_match = None
    best_score = -1

    for v in voicings:
        score = 0
        for i in range(6):
            string = 6 - i
            vfret = v['frets'][i]
            actual_fret = (vfret + v['baseFret'] - 1) if vfret > 0 else vfret
            if string in note_frets:
                if vfret == 0 and note_frets[string] == 0:
                    score += 2
                elif actual_fret == note_frets[string]:
                    score += 2
        if score > best_score:
            best_score = score
            best_match = v

    if best_match and best_score > 0:
        result = {}
        for i in range(6):
            string = 6 - i
            fret = best_match['frets'][i]
            finger = best_match['fingers'][i]
            if fret > 0 and finger > 0:
                actual_fret = fret + best_match['baseFret'] - 1
                result[(string, actual_fret)] = finger
            elif fret == 0:
                result[(string, 0)] = 0
        return result
    return {}


def _match_mass(note_frets: dict, voicings: list) -> dict:
    """mass DB (chord-collection形式、絶対フレット) からベストマッチを検索"""
    best_match = None
    best_score = -1

    for v in voicings:
        score = 0
        for i in range(6):
            string = 6 - i
            fret = v['frets_abs'][i]
            if string in note_frets:
                if fret == note_frets[string]:
                    score += 2
                elif fret == 0 and note_frets[string] == 0:
                    score += 2
        if score > best_score:
            best_score = score
            best_match = v

    if best_match and best_score > 0:
        result = {}
        for i in range(6):
            string = 6 - i
            fret = best_match['frets_abs'][i]
            finger = best_match['fingers'][i]
            if fret > 0 and finger > 0:
                result[(string, fret)] = finger
            elif fret == 0:
                result[(string, 0)] = 0
        return result
    return {}


def lookup_chord_fingers(chord_name: str, notes: list) -> dict:
    """コード名とノート群から最適なボイシングの指番号を検索。
    
    2層検索: curated DB (3,283) → mass DB (167,011)
    Returns: {(string, fret): finger} or empty
    """
    if not chord_name:
        return {}

    root, suffix = _parse_chord_name(chord_name)
    if not root:
        return {}

    # suffix正規化
    suffix_map = {
        '': 'major', 'maj': 'major', 'min': 'minor', 'm': 'minor',
        '7': '7', 'maj7': 'maj7', 'm7': 'minor7',
        'dim': 'dim', 'aug': 'aug', 'sus4': 'sus4', 'sus2': 'sus2',
    }

    note_frets = {}
    for n in notes:
        s = n.get('string', 0)
        f = n.get('fret', 0)
        if s > 0:
            note_frets[s] = f

    # --- 1st: curated DB ---
    curated = _load_curated_db()
    norm_suffix = suffix_map.get(suffix, suffix)
    db_key = f"{root}_{norm_suffix}"

    voicings = curated.get(db_key, [])
    if not voicings:
        alt = _ENHARMONIC.get(root)
        if alt:
            voicings = curated.get(f"{alt}_{norm_suffix}", [])

    if voicings:
        result = _match_curated(note_frets, voicings)
        if result:
            return result

    # --- 2nd: mass DB ---
    mass = _load_mass_db()
    # chord-collectionのキーは "C", "Cm", "C7" 等
    mass_key = chord_name
    voicings = mass.get(mass_key, [])
    if not voicings:
        # root + suffix の組み合わせ試行
        for try_key in [f"{root}{suffix}", f"{root}", chord_name]:
            voicings = mass.get(try_key, [])
            if voicings:
                break
    if not voicings:
        alt = _ENHARMONIC.get(root)
        if alt:
            for try_key in [f"{alt}{suffix}", f"{alt}"]:
                voicings = mass.get(try_key, [])
                if voicings:
                    break

    if voicings:
        result = _match_mass(note_frets, voicings)
        if result:
            return result

    return {}


def lookup_scale_fingers(key: str, fret: int, string: int,
                         position: int = 0) -> int:
    """キーとポジションからスケール音の標準指番号を返す。"""
    if fret == 0:
        return 0
    return fret_to_finger_in_position(fret, position)


# テクニック別の指制約
TECHNIQUE_FINGER_RULES = {
    "h": "ascending",
    "p": "descending",
    "/": "same",
    "\\": "same",
}

# 弦ごとの指ペナルティ（データ駆動）
STRING_FINGER_PENALTY = {
    (6, 4): 8.0, (6, 3): 5.0,
    (5, 4): 5.0,
    (1, 2): 5.0,
}

# ---------------------------------------------------------------------------
# Pentatonic minor boxes (Am pentatonic, 5 CAGED boxes)
# Each box: list of (string, offset_from_box_start, finger)
#   string: 1–6 (1=high E, 6=low E)
#   offset_from_box_start: fret offset relative to the box's starting fret
#   finger: 1=index, 2=middle, 3=ring, 4=pinky, 0=open
# Standard fingerings matching classical/jazz guitar pedagogy.
# ---------------------------------------------------------------------------
PENTATONIC_MINOR_BOXES: dict[int, list[tuple[int, int, int]]] = {
    # Box 1 — "E shape": root on 6th string (e.g. Am at fret 5)
    1: [
        (6, 0, 1), (6, 3, 4),
        (5, 0, 1), (5, 3, 4),
        (4, 0, 1), (4, 2, 3),
        (3, 0, 1), (3, 2, 3),
        (2, 0, 1), (2, 3, 4),
        (1, 0, 1), (1, 3, 4),
    ],
    # Box 2 — "D shape": root on 4th string
    2: [
        (6, 0, 1), (6, 2, 3),
        (5, 0, 1), (5, 2, 3),
        (4, 0, 1), (4, 2, 4),
        (3, 0, 1), (3, 2, 3),
        (2, 0, 1), (2, 2, 3),
        (1, 0, 1), (1, 2, 3),
    ],
    # Box 3 — "C shape"
    3: [
        (6, 0, 1), (6, 3, 4),
        (5, 0, 1), (5, 2, 3),
        (4, 0, 1), (4, 2, 3),
        (3, 0, 1), (3, 2, 4),
        (2, 0, 1), (2, 3, 4),
        (1, 0, 1), (1, 3, 4),
    ],
    # Box 4 — "A shape": root on 5th string
    4: [
        (6, 0, 1), (6, 2, 3),
        (5, 0, 1), (5, 2, 3),
        (4, 0, 1), (4, 2, 4),
        (3, 0, 2), (3, 2, 4),
        (2, 0, 1), (2, 2, 3),
        (1, 0, 1), (1, 2, 3),
    ],
    # Box 5 — "G shape": root on 3rd string
    5: [
        (6, 0, 1), (6, 2, 4),
        (5, 0, 1), (5, 3, 4),
        (4, 0, 1), (4, 2, 3),
        (3, 0, 1), (3, 2, 3),
        (2, 0, 1), (2, 3, 4),
        (1, 0, 1), (1, 2, 4),
    ],
}

# ---------------------------------------------------------------------------
# Major scale — 3-note-per-string (3NPS), 7 positions
# Each position: list of (string, offset_from_position, finger)
# Fingerings follow the Satriani / modern-rock 3NPS convention:
#   1-2-4 for whole-whole, 1-3-4 for whole-half, 1-2-3 for half-whole
# ---------------------------------------------------------------------------
MAJOR_SCALE_3NPS: dict[int, list[tuple[int, int, int]]] = {
    # Position 1 (Ionian) — e.g. C major starting fret 8 on 6th string
    1: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),   # W W
        (5, 0, 1), (5, 2, 2), (5, 4, 4),   # W W
        (4, 0, 1), (4, 1, 2), (4, 3, 4),   # H W
        (3, 0, 1), (3, 2, 3), (3, 4, 4),   # W W  (shift for 3rd-string offset)
        (2, 0, 1), (2, 1, 2), (2, 3, 4),   # H W
        (1, 0, 1), (1, 2, 2), (1, 4, 4),   # W W
    ],
    # Position 2 (Dorian)
    2: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),   # W W
        (5, 0, 1), (5, 1, 2), (5, 3, 4),   # H W
        (4, 0, 1), (4, 2, 3), (4, 4, 4),   # W W
        (3, 0, 1), (3, 1, 2), (3, 3, 4),   # H W
        (2, 0, 1), (2, 2, 2), (2, 4, 4),   # W W
        (1, 0, 1), (1, 2, 2), (1, 3, 3),   # W H
    ],
    # Position 3 (Phrygian)
    3: [
        (6, 0, 1), (6, 1, 2), (6, 3, 4),   # H W
        (5, 0, 1), (5, 2, 3), (5, 4, 4),   # W W
        (4, 0, 1), (4, 1, 2), (4, 3, 4),   # H W
        (3, 0, 1), (3, 2, 2), (3, 4, 4),   # W W
        (2, 0, 1), (2, 2, 2), (2, 3, 3),   # W H
        (1, 0, 1), (1, 2, 2), (1, 4, 4),   # W W
    ],
    # Position 4 (Lydian)
    4: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),   # W W
        (5, 0, 1), (5, 1, 2), (5, 3, 4),   # H W
        (4, 0, 1), (4, 2, 2), (4, 4, 4),   # W W
        (3, 0, 1), (3, 2, 2), (3, 3, 3),   # W H
        (2, 0, 1), (2, 2, 2), (2, 4, 4),   # W W
        (1, 0, 1), (1, 1, 2), (1, 3, 4),   # H W
    ],
    # Position 5 (Mixolydian)
    5: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),   # W W
        (5, 0, 1), (5, 2, 3), (5, 4, 4),   # W W
        (4, 0, 1), (4, 2, 2), (4, 3, 3),   # W H
        (3, 0, 1), (3, 2, 2), (3, 4, 4),   # W W
        (2, 0, 1), (2, 1, 2), (2, 3, 4),   # H W
        (1, 0, 1), (1, 2, 2), (1, 4, 4),   # W W
    ],
    # Position 6 (Aeolian / natural minor)
    6: [
        (6, 0, 1), (6, 2, 2), (6, 3, 3),   # W H
        (5, 0, 1), (5, 2, 2), (5, 4, 4),   # W W
        (4, 0, 1), (4, 1, 2), (4, 3, 4),   # H W
        (3, 0, 1), (3, 2, 2), (3, 4, 4),   # W W
        (2, 0, 1), (2, 1, 2), (2, 3, 4),   # H W
        (1, 0, 1), (1, 2, 2), (1, 3, 3),   # W H
    ],
    # Position 7 (Locrian)
    7: [
        (6, 0, 1), (6, 1, 2), (6, 3, 4),   # H W
        (5, 0, 1), (5, 2, 2), (5, 4, 4),   # W W
        (4, 0, 1), (4, 2, 3), (4, 3, 4),   # W H
        (3, 0, 1), (3, 2, 2), (3, 4, 4),   # W W
        (2, 0, 1), (2, 1, 2), (2, 3, 4),   # H W
        (1, 0, 1), (1, 2, 2), (1, 4, 4),   # W W
    ],
}

# ---------------------------------------------------------------------------
# Natural minor scale — 3NPS, 7 positions
# The natural minor is the Aeolian mode. We rotate the major 3NPS patterns
# so that position 1 starts from the minor root (= major position 6).
# Providing explicit data for clarity and independent maintenance.
# ---------------------------------------------------------------------------
NATURAL_MINOR_SCALE_3NPS: dict[int, list[tuple[int, int, int]]] = {
    # Position 1 (Aeolian root)
    1: [
        (6, 0, 1), (6, 2, 2), (6, 3, 3),
        (5, 0, 1), (5, 2, 2), (5, 4, 4),
        (4, 0, 1), (4, 1, 2), (4, 3, 4),
        (3, 0, 1), (3, 2, 2), (3, 4, 4),
        (2, 0, 1), (2, 1, 2), (2, 3, 4),
        (1, 0, 1), (1, 2, 2), (1, 3, 3),
    ],
    # Position 2
    2: [
        (6, 0, 1), (6, 1, 2), (6, 3, 4),
        (5, 0, 1), (5, 2, 2), (5, 4, 4),
        (4, 0, 1), (4, 2, 3), (4, 3, 4),
        (3, 0, 1), (3, 2, 2), (3, 4, 4),
        (2, 0, 1), (2, 2, 2), (2, 4, 4),
        (1, 0, 1), (1, 2, 2), (1, 4, 4),
    ],
    # Position 3
    3: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),
        (5, 0, 1), (5, 2, 2), (5, 3, 3),
        (4, 0, 1), (4, 2, 2), (4, 4, 4),
        (3, 0, 1), (3, 2, 2), (3, 4, 4),
        (2, 0, 1), (2, 2, 2), (2, 4, 4),
        (1, 0, 1), (1, 2, 2), (1, 4, 4),
    ],
    # Position 4
    4: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),
        (5, 0, 1), (5, 1, 2), (5, 3, 4),
        (4, 0, 1), (4, 2, 3), (4, 4, 4),
        (3, 0, 1), (3, 2, 2), (3, 4, 4),
        (2, 0, 1), (2, 2, 2), (2, 4, 4),
        (1, 0, 1), (1, 1, 2), (1, 3, 4),
    ],
    # Position 5
    5: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),
        (5, 0, 1), (5, 2, 3), (5, 4, 4),
        (4, 0, 1), (4, 2, 2), (4, 4, 4),
        (3, 0, 1), (3, 2, 2), (3, 3, 3),
        (2, 0, 1), (2, 2, 2), (2, 4, 4),
        (1, 0, 1), (1, 1, 2), (1, 3, 4),
    ],
    # Position 6
    6: [
        (6, 0, 1), (6, 2, 2), (6, 4, 4),
        (5, 0, 1), (5, 2, 2), (5, 4, 4),
        (4, 0, 1), (4, 2, 2), (4, 3, 3),
        (3, 0, 1), (3, 2, 2), (3, 4, 4),
        (2, 0, 1), (2, 1, 2), (2, 3, 4),
        (1, 0, 1), (1, 2, 2), (1, 4, 4),
    ],
    # Position 7
    7: [
        (6, 0, 1), (6, 2, 2), (6, 3, 3),
        (5, 0, 1), (5, 2, 2), (5, 4, 4),
        (4, 0, 1), (4, 2, 3), (4, 3, 4),
        (3, 0, 1), (3, 2, 2), (3, 4, 4),
        (2, 0, 1), (2, 1, 2), (2, 3, 4),
        (1, 0, 1), (1, 2, 2), (1, 3, 3),
    ],
}

# ---------------------------------------------------------------------------
# Arpeggio patterns — standard shapes with pedagogical fingerings
# Each pattern: list of (string, offset_from_root, finger)
# ---------------------------------------------------------------------------
ARPEGGIO_PATTERNS: dict[str, list[tuple[int, int, int]]] = {
    # --- Major triad ---
    "major_root": [
        # Root-position major triad, root on 6th string (E-shape barre base)
        (6, 0, 1), (5, 2, 3), (4, 2, 4),
        (3, 1, 2), (2, 0, 1), (1, 0, 1),
    ],
    "major_1st_inv": [
        # 1st inversion, root on 4th string
        (5, 0, 1), (4, 2, 3),
        (3, 1, 2), (2, 0, 1), (1, 0, 1),
    ],
    "major_2nd_inv": [
        # 2nd inversion, root on 3rd string
        (5, 0, 1), (4, 1, 1),
        (3, 0, 1), (2, 1, 2), (1, 0, 1),
    ],
    # --- Minor triad ---
    "minor_root": [
        # Root-position minor triad, root on 6th string (Em-shape barre base)
        (6, 0, 1), (5, 2, 3), (4, 2, 4),
        (3, 0, 1), (2, 0, 1), (1, 0, 1),
    ],
    "minor_1st_inv": [
        (5, 0, 1), (4, 2, 3),
        (3, 0, 1), (2, 1, 2), (1, 0, 1),
    ],
    "minor_2nd_inv": [
        (5, 0, 1), (4, 1, 1),
        (3, 0, 1), (2, 0, 1), (1, 1, 2),
    ],
    # --- Dominant 7th ---
    "dom7_root": [
        # Root-position dominant 7th, root on 6th string (E7-shape barre)
        (6, 0, 1), (5, 2, 3), (4, 0, 1),
        (3, 1, 2), (2, 0, 1), (1, 0, 1),
    ],
    "dom7_drop2": [
        # Drop-2 voicing, root on 5th string
        (5, 0, 1), (4, 2, 2), (3, 1, 1),
        (2, 3, 4), (1, 0, 1),
    ],
    "dom7_drop3": [
        # Drop-3 voicing, root on 6th string
        (6, 0, 1), (4, 1, 1), (3, 2, 2),
        (2, 1, 1), (1, 0, 1),
    ],
}

# --- すべてのスケール/アルペジオパターンを統合インデックスに ---
_ALL_SCALE_BOXES: list[tuple[str, list[tuple[int, int, int]]]] = []


def _build_scale_index() -> list[tuple[str, list[tuple[int, int, int]]]]:
    """Lazily build a flat index of all scale/arpeggio patterns for matching."""
    global _ALL_SCALE_BOXES
    if _ALL_SCALE_BOXES:
        return _ALL_SCALE_BOXES

    for box_num, notes in PENTATONIC_MINOR_BOXES.items():
        _ALL_SCALE_BOXES.append((f"pentatonic_minor_box{box_num}", notes))
    for pos_num, notes in MAJOR_SCALE_3NPS.items():
        _ALL_SCALE_BOXES.append((f"major_3nps_pos{pos_num}", notes))
    for pos_num, notes in NATURAL_MINOR_SCALE_3NPS.items():
        _ALL_SCALE_BOXES.append((f"natural_minor_3nps_pos{pos_num}", notes))
    for name, notes in ARPEGGIO_PATTERNS.items():
        _ALL_SCALE_BOXES.append((f"arpeggio_{name}", notes))

    return _ALL_SCALE_BOXES


def match_scale_box(notes: list, key: str = None) -> list:
    """Try to match a sequence of notes to a known scale box pattern.

    Args:
        notes: List of note dicts with 'string', 'fret', 'pitch' keys
        key: Optional detected key (e.g. 'Am', 'C', 'G')

    Returns:
        List of (note_index, suggested_finger, box_name) or empty list
    """
    if len(notes) < 4:
        return []

    # Filter to fretted notes only and build (index, string, fret) triples
    fretted: list[tuple[int, int, int]] = []
    for i, n in enumerate(notes):
        s = n.get('string', 0)
        f = n.get('fret', 0)
        if s > 0 and f > 0:
            fretted.append((i, s, f))

    if len(fretted) < 4:
        return []

    # Extract (string, fret) pairs for the fretted notes
    sf_pairs = [(s, f) for _, s, f in fretted]

    # Determine the fret range to limit transposition search
    min_fret = min(f for _, f in sf_pairs)
    max_fret = max(f for _, f in sf_pairs)

    all_boxes = _build_scale_index()
    best_result: list[tuple[int, int, str]] = []
    best_match_ratio = 0.0

    for box_name, pattern in all_boxes:
        # Determine max offset in this pattern to compute transposition range
        max_pattern_offset = max(off for _, off, _ in pattern) if pattern else 0

        # Build a lookup: (string, offset) -> finger
        pattern_lookup: dict[tuple[int, int], int] = {}
        for s, off, finger in pattern:
            pattern_lookup[(s, off)] = finger

        # Try all transpositions (shift = the absolute fret that offset 0 maps to)
        for shift in range(max(0, min_fret - max_pattern_offset), max_fret + 1):
            matched: list[tuple[int, int, str]] = []
            for idx, s, f in fretted:
                offset = f - shift
                if (s, offset) in pattern_lookup:
                    matched.append((idx, pattern_lookup[(s, offset)], box_name))

            match_ratio = len(matched) / len(fretted) if fretted else 0
            if match_ratio >= 0.80 and match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_result = matched

    return best_result


# --- 手動スケールパターン（Segovia式） ---
SCALE_PATTERNS = {
    ("Em", 0): [
        (6,0,0), (6,2,2), (6,3,3),
        (5,0,0), (5,2,2), (5,3,3),
        (4,0,0), (4,2,2),
        (3,0,0), (3,2,2),
        (2,0,0), (2,1,1), (2,3,3),
        (1,0,0), (1,2,2), (1,3,3),
    ],
    ("C", 0): [
        (6,0,0), (6,1,1), (6,3,3),
        (5,0,0), (5,2,2), (5,3,3),
        (4,0,0), (4,2,2), (4,3,4),
        (3,0,0), (3,2,2),
        (2,0,0), (2,1,1), (2,3,3),
        (1,0,0), (1,1,1), (1,3,3),
    ],
    ("G", 0): [
        (6,0,0), (6,2,1), (6,3,2),
        (5,0,0), (5,2,2), (5,3,4),
        (4,0,0), (4,2,2), (4,4,4),
        (3,0,0), (3,2,2),
        (2,0,0), (2,1,1), (2,3,3),
        (1,0,0), (1,2,2), (1,3,3),
    ],
}

# コードフォーム（手動 — 最終フォールバック）
CHORD_FORMS = {
    "C":  [(5,3,3), (4,2,2), (3,0,0), (2,1,1), (1,0,0)],
    "Am": [(5,0,0), (4,2,2), (3,2,3), (2,1,1), (1,0,0)],
    "Em": [(5,2,2), (4,2,3), (3,0,0), (2,0,0), (1,0,0)],
    "E":  [(6,0,0), (5,2,2), (4,2,3), (3,1,1), (2,0,0), (1,0,0)],
    "Dm": [(4,0,0), (3,2,3), (2,3,4), (1,1,1)],
    "D":  [(4,0,0), (3,2,1), (2,3,3), (1,2,2)],
    "G":  [(6,3,2), (5,2,1), (4,0,0), (3,0,0), (2,0,0), (1,3,3)],
    "A":  [(5,0,0), (4,2,1), (3,2,2), (2,2,3), (1,0,0)],
    "F":  [(6,1,1), (5,3,3), (4,3,4), (3,2,2), (2,1,1), (1,1,1)],
}
