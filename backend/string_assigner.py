"""
string_assigner.py — 弦・フレット最適割り当て (研究論文ベース改善版)
=====================================================================
MIDIノート番号を (弦, フレット) に変換する。

アルゴリズム基盤:
- Viterbi DP: フレーズ全体で最適パス探索 (ISMIR "Minimax Approach to Guitar Fingering")
- 多属性コスト関数: 位置/遷移/人間工学/音色の4カテゴリ (Bontempi "From MIDI to Rich Tablatures")
- 指モデリング: ポジション依存フレットスパン (KTH "Guitar Fingering for Music Score")
- Minimax後処理: 最大難度の1手を回避 (ISMIR Minimax Viterbi)
- ポジションウィンドウ: 1st/5th/7th等のポジション概念 (gtrsnipe positional score)

ギター指板理論:
- 1ポジション = 4フレット幅 (人差指〜小指)
- フレットスパン > 4 は物理的に弾けない (ローポジションでは3)
- 2弦-3弦間は4半音 (他は5半音)
- sweet spot: 0-7フレット
- 同一ピッチの複数ポジション → Viterbi DPで全体最適を選択
- 人間運指選好マップ: IDMTの人間演奏データから学習したポジション選好を統合
"""

from typing import List, Tuple, Optional, Dict
from itertools import product as iter_product
from collections import Counter
import math
import json
import os
import numpy as np

# =============================================================================
# 弦分類器 (String Classifier CNN) — 音声CQT特徴量から弦を推定
# =============================================================================

_STRING_CLASSIFIER = None  # lazy-loaded
_STRING_CLASSIFIER_CQT_CACHE = {}  # audio_path -> CQT array

def _load_string_classifier():
    """弦分類器CNNモデルをlazy-load。"""
    global _STRING_CLASSIFIER
    if _STRING_CLASSIFIER is not None:
        return _STRING_CLASSIFIER
    
    model_path = os.path.join(os.path.dirname(__file__), "string_classifier.pth")
    if not os.path.exists(model_path):
        _STRING_CLASSIFIER = False  # モデルなし
        return False
    
    try:
        import torch
        from string_classifier import StringClassifierCNN, N_BINS, CONTEXT_FRAMES
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StringClassifierCNN(n_bins=N_BINS, n_frames=CONTEXT_FRAMES).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        _STRING_CLASSIFIER = {'model': model, 'device': device}
        print(f"[string_assigner] 弦分類器ロード完了 (device={device})")
        return _STRING_CLASSIFIER
    except Exception as e:
        print(f"[string_assigner] 弦分類器ロード失敗: {e}")
        _STRING_CLASSIFIER = False
        return False


def _compute_cqt_cached(audio_path: str) -> Optional[np.ndarray]:
    """CQTスペクトログラムをキャッシュ付きで計算。"""
    if audio_path in _STRING_CLASSIFIER_CQT_CACHE:
        return _STRING_CLASSIFIER_CQT_CACHE[audio_path]
    
    try:
        import librosa
        import soundfile as sf
        from string_classifier import SR, HOP_LENGTH, N_BINS
        
        y, sr_orig = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr_orig != SR:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
        
        cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH,
                                  n_bins=N_BINS, bins_per_octave=12))
        cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt = (cqt + 80) / 80
        cqt = np.clip(cqt, 0, 1)
        
        _STRING_CLASSIFIER_CQT_CACHE[audio_path] = cqt
        return cqt
    except Exception as e:
        print(f"[string_assigner] CQT計算エラー: {e}")
        return None


def _predict_string_probs(audio_path: str, onset_time: float,
                           midi_pitch: int) -> Optional[Dict[int, float]]:
    """弦分類器で弦確率を予測。Returns: {1: prob, ..., 6: prob} or None."""
    clf = _load_string_classifier()
    if not clf:
        return None
    
    cqt = _compute_cqt_cached(audio_path)
    if cqt is None:
        return None
    
    try:
        import torch
        from string_classifier import SR, HOP_LENGTH, CONTEXT_FRAMES
        
        frame_idx = int(onset_time * SR / HOP_LENGTH)
        half_ctx = CONTEXT_FRAMES // 2
        
        if frame_idx - half_ctx < 0 or frame_idx + half_ctx >= cqt.shape[1]:
            return None
        
        patch = cqt[:, frame_idx - half_ctx:frame_idx + half_ctx + 1]
        device = clf['device']
        patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
        pitch_tensor = torch.FloatTensor([(midi_pitch - 40) / 45.0]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = clf['model'](patch_tensor, pitch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        return {s + 1: float(probs[s]) for s in range(6)}
    except Exception:
        return None

# =============================================================================
# 音楽理論エンジン: 典型フォームDB + コード情報統合
# (坂井論文 2024「主旋律と和音を同時に演奏するソロギターのためのタブ譜生成」準拠)
# =============================================================================

# 典型フォームDBのロード
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


# コード構成音マッピング（ルートからの半音数）
_CHORD_INTERVALS = {
    "major": [0, 4, 7], "minor": [0, 3, 7], "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11], "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9], "aug": [0, 4, 8], "sus4": [0, 5, 7],
    "sus2": [0, 2, 7], "m7b5": [0, 3, 6, 10], "6": [0, 4, 7, 9],
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
    
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    QUALITY_SUFFIX = {
        "major": "", "minor": "m", "7": "7", "m7": "m7", "maj7": "maj7",
        "dim": "dim", "dim7": "dim7", "aug": "aug", "sus4": "sus4", "sus2": "sus2",
        "m7b5": "m7b5", "6": "6",
    }
    db_key = NOTE_NAMES[root_pc] + QUALITY_SUFFIX.get(quality, "")
    
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
    
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    QUALITY_SUFFIX = {
        "major": "", "minor": "m", "7": "7", "m7": "m7", "maj7": "maj7",
        "dim": "dim", "dim7": "dim7", "aug": "aug", "sus4": "sus4", "sus2": "sus2",
        "m7b5": "m7b5", "6": "6",
    }
    db_key = NOTE_NAMES[root_pc] + QUALITY_SUFFIX.get(quality, "")
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
    if 0 <= idx < len(tuning):
        pc = (tuning[idx] + f) % 12
        if pc in chord_pcs:
            return -3.0  # コード構成音だがフォーム外 → 小さいボーナス
    
    return 0.0  # フォーム外・非構成音 → ペナルティなし


# 標準チューニング: 6弦→1弦 (低→高)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

# ===== ソロギター変則チューニング大辞典 =====
# MIDI note numbers: C0=0, C1=12, C2=24, ...
# E2=40, F2=41, F#2=42, G2=43, Ab2=44, A2=45, Bb2=46, B2=47
# C3=48, C#3=49, D3=50, Eb3=51, E3=52, F3=53, F#3=54, G3=55
# Ab3=56, A3=57, Bb3=58, B3=59, C4=60, C#4=61, D4=62, Eb4=63, E4=64

TUNINGS = {
    # --- スタンダード系 ---
    "standard":       [40, 45, 50, 55, 59, 64],  # E A D G B E
    "half_down":      [39, 44, 49, 54, 58, 63],  # Eb Ab Db Gb Bb Eb
    "full_down":      [38, 43, 48, 53, 57, 62],  # D G C F A D
    "1half_down":     [37, 42, 47, 52, 56, 61],  # Db Gb B E Ab Db

    # --- Drop系 ---
    "drop_d":         [38, 45, 50, 55, 59, 64],  # D A D G B E
    "drop_c#":        [37, 44, 49, 54, 58, 63],  # C# Ab Db Gb Bb Eb
    "drop_c":         [36, 43, 48, 53, 57, 62],  # C G C F A D
    "drop_b":         [35, 42, 47, 52, 56, 61],  # B Gb B E Ab Db
    "double_drop_d":  [38, 45, 50, 55, 59, 62],  # D A D G B D

    # --- DADGAD系 (ケルティック・押尾コータロー頻用) ---
    "dadgad":         [38, 45, 50, 55, 57, 62],  # D A D G A D
    "dadgac":         [38, 45, 50, 55, 57, 60],  # D A D G A C (押尾「twilight」等)
    "dadgae":         [38, 45, 50, 55, 57, 64],  # D A D G A E
    "dadead":         [38, 45, 50, 52, 57, 62],  # D A D E A D
    "cgdgad":         [36, 43, 50, 55, 57, 62],  # C G D G A D
    "cgdgbd":         [36, 43, 50, 55, 59, 62],  # C G D G B D
    "dadf#ad":        [38, 45, 50, 54, 57, 62],  # D A D F# A D (Open D variant)
    "daddad":         [38, 45, 50, 50, 57, 62],  # D A D D A D

    # --- Open Major系 ---
    "open_d":         [38, 45, 50, 54, 57, 62],  # D A D F# A D
    "open_e":         [40, 47, 52, 56, 59, 64],  # E B E G# B E
    "open_g":         [38, 43, 50, 55, 59, 62],  # D G D G B D
    "open_a":         [40, 45, 52, 57, 61, 64],  # E A E A C# E
    "open_c":         [36, 43, 48, 55, 60, 64],  # C G C G C E
    "open_c6":        [36, 45, 48, 55, 60, 64],  # C A C G C E

    # --- Open Minor系 ---
    "open_dm":        [38, 45, 50, 53, 57, 62],  # D A D F A D
    "open_em":        [40, 47, 52, 55, 59, 64],  # E B E G B E
    "open_gm":        [38, 43, 50, 55, 58, 62],  # D G D G Bb D
    "open_am":        [40, 45, 52, 57, 60, 64],  # E A E A C E
    "open_cm":        [36, 43, 48, 55, 60, 63],  # C G C G C Eb

    # --- Nashville / New Standard ---
    "nashville":      [52, 57, 62, 67, 71, 76],  # E3 A3 D4 G4 B4 E5 (高音)
    "new_standard":   [36, 43, 50, 57, 62, 67],  # C G D A E B (Fripp)

    # --- 押尾コータロー特有 ---
    "oshio_wind":     [38, 45, 50, 55, 57, 62],  # DADGAD (Wind Song等)
    "oshio_fight":    [38, 43, 50, 55, 59, 62],  # DGDGBD = Open G (Fight!等)
    "oshio_landscape":[36, 43, 50, 55, 57, 62],  # CGDGAD (Landscape等)

    # --- Andy McKee / Antoine Dufour / Michael Hedges ---
    "cgcgce":         [36, 43, 48, 55, 60, 64],  # C G C G C E = Open C
    "cgcgcg":         [36, 43, 48, 55, 60, 67],  # C G C G C G
    "bebebe":         [35, 40, 47, 52, 59, 64],  # B E B E B E
    "dadaad":         [38, 45, 50, 57, 57, 62],  # D A D A A D
    "cgdgbe":         [36, 43, 50, 55, 59, 64],  # C G D G B E

    # --- Eb系 (半音下げ) ---
    "eb_drop_db":     [37, 44, 49, 54, 58, 63],  # Db Ab Db Gb Bb Eb
}

# チューニング名→表示名マッピング
TUNING_DISPLAY_NAMES = {
    "standard": "スタンダード (EADGBE)",
    "half_down": "半音下げ (Eb)",
    "full_down": "全音下げ (D)",
    "drop_d": "Drop D",
    "drop_c": "Drop C",
    "double_drop_d": "Double Drop D",
    "dadgad": "DADGAD",
    "dadgac": "DADGAC",
    "open_d": "Open D",
    "open_e": "Open E",
    "open_g": "Open G",
    "open_a": "Open A",
    "open_c": "Open C",
    "open_dm": "Open Dm",
    "open_em": "Open Em",
    "open_gm": "Open Gm",
    "open_am": "Open Am",
    "cgcgce": "CGCGCE",
    "cgdgad": "CGDGAD",
}


def guess_tuning(notes: list, top_n: int = 3) -> list:
    """
    検出されたノートの最低音パターンからチューニングを推定する。
    各チューニングについて、検出された最低音がオープン弦音と一致する確率を計算。

    Returns: [(tuning_name, score), ...] 上位top_n件
    """
    if not notes:
        return [("standard", 1.0)]

    # 最低音を収集（曲の冒頭30秒と全体）
    pitches = sorted(set(n["pitch"] for n in notes))
    lowest_10 = pitches[:10]  # 最低音域の10音

    results = []
    for name, tuning in TUNINGS.items():
        score = 0.0
        open_pitches = set(tuning)

        # 検出された最低音がオープン弦と一致するか
        for p in lowest_10:
            if p in open_pitches:
                score += 2.0
            # オクターブ上のオープン弦音にも一致
            elif (p - 12) in open_pitches:
                score += 0.5

        # 最低音がチューニングの6弦と一致するとボーナス
        if pitches[0] == tuning[0]:
            score += 5.0
        elif pitches[0] == tuning[0] + 12:
            score += 2.0

        # 全体のピッチがチューニング範囲内か
        min_tuning = min(tuning)
        if pitches[0] >= min_tuning:
            score += 1.0

        results.append((name, score))

    results.sort(key=lambda x: -x[1])
    return results[:top_n]

MAX_FRET = 19  # アコースティックギターの実用フレット範囲


# =============================================================================
# ポジション & 候補列挙
# =============================================================================

def get_possible_positions(pitch: int, tuning: List[int] = None,
                           max_fret: int = MAX_FRET) -> List[Tuple[int, int]]:
    """
    Returns all possible (string, fret) positions for a given MIDI pitch.

    Parameters
    ----------
    pitch : int
        MIDI note number.
    tuning : list[int]
        Open string MIDI notes [6th, 5th, 4th, 3rd, 2nd, 1st].
    max_fret : int
        Maximum fret number.

    Returns
    -------
    list of (string_number, fret_number)
        string_number: 1-6 (1=highest E, 6=lowest E)
        fret_number: 0-max_fret
    """
    if tuning is None:
        tuning = STANDARD_TUNING

    positions = []
    for i, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            string_num = 6 - i   # 6弦=index 0 → string 6
            positions.append((string_num, fret))

    return positions


# =============================================================================
# 多属性コスト関数 (Bontempi "From MIDI to Rich Tablatures" ベース)
# =============================================================================
# コストが低いほど良い。全てのコストは非負。

# --- 重みパラメータ ---
WEIGHTS = {
    # 位置コスト
    "w_fret_height":          0.05,   # フレット高コスト (PDL最適化: 0.8→0.05 中フレット許容)
    "w_high_fret_extra":      4.5,    # 9f超追加コスト (PDL最適化: 3.0→4.5 極端なハイポジ回避)
    "w_low_string_high_fret": 1.5,    # 低弦(4-6弦)ハイフレット倍率
    "w_sweet_spot_bonus":    -1.0,    # sweet spot (0-9f) ボーナス (PDL最適化: -1.5→-1.0)

    # 遷移コスト（ポジション連続性を強く重視）
    "w_movement":            15.0,    # ポジション移動コスト (フレット差に比例)
    "w_position_shift":      50.0,    # ポジション跨ぎ追加コスト (4f超の移動)
    "w_string_switch":        2.0,    # 弦切り替えコスト (弦距離に比例)
    "w_same_string_repeat":   5.5,    # ⑧ 右手PIMA: 同弦連打ペナルティ (PDL最適化: 5.0→5.5)

    # 人間工学コスト
    "w_fret_span":          100.0,    # 和音フレットスパンコスト
    "w_unplayable":       10000.0,    # 物理的に弾けない配置
    "w_adjacent_stretch":    30.0,    # ⑨ 隣接弦ストレッチペナルティ (3f超)
    "w_too_many_fingers":  5000.0,    # ⑨ 4音超の同時押弦ペナルティ (バレーなし)

    # 音色コスト
    "w_open_string_bonus":   -1.0,    # 開放弦ボーナス（微量—過剰優遍防止）
    "w_open_match_bonus":    -5.0,    # 開放弦でしか出せない音のボーナス
    "w_barre_bonus":         -5.0,    # バレーコードボーナス (per extra string)

    # ⑦ フィンガースタイル弦域分離 (SMC Fingerstyle論文)
    "w_bass_low_string":   -20.0,    # ベース音(最低ピッチ)が低弦(4-6弦)ボーナス
    "w_melody_high_string":-15.0,    # メロディ音(最高ピッチ)が高弦(1-3弦)ボーナス
    "w_bass_wrong_string":  25.0,    # ベース音が高弦(1-3弦)にいるペナルティ
    # 人間運指選好 (IDMT human fingering)
    "w_human_pref_bonus":   -15.0,   # 人間が好むポジションへのボーナス
}

# =============================================================================
# 人間運指選好マップ (IDMT-SMT-V2 由来)
# =============================================================================
_HUMAN_PREF_MAP = None

def _load_human_preference():
    """人間運指選好マップをロード (初回のみ)"""
    global _HUMAN_PREF_MAP
    if _HUMAN_PREF_MAP is not None:
        return _HUMAN_PREF_MAP

    pref_path = os.path.join(os.path.dirname(__file__), 'human_position_preference.json')
    if os.path.exists(pref_path):
        with open(pref_path, 'r', encoding='utf-8') as f:
            _HUMAN_PREF_MAP = json.load(f)
        print(f"[string_assigner] 人間運指選好マップ: {len(_HUMAN_PREF_MAP)} pitches loaded")
    else:
        _HUMAN_PREF_MAP = {}
    return _HUMAN_PREF_MAP


def _human_preference_cost(pitch: int, s: int, f: int) -> float:
    """
    人間運指選好コスト: IDMTの人間が選んだポジションにボーナスを付与。
    人間が高頻度で選ぶポジション → 低コスト（ボーナス）
    人間が選ばないポジション → コスト0（ペナルティなし）
    """
    pref = _load_human_preference()
    if not pref:
        return 0.0

    pitch_data = pref.get(str(pitch))
    if not pitch_data:
        return 0.0

    prob = pitch_data.get('prob', {})
    key = f"{s}_{f}"
    p = prob.get(key, 0.0)

    if p > 0:
        # 人間がこのポジションを選んだ確率に応じたボーナス
        return WEIGHTS["w_human_pref_bonus"] * p
    return 0.0

# --- ポジション定義 ---
# ギタリストは「ポジション」単位で指板を認識する
POSITION_WIDTH = 4  # 1ポジションのフレット幅（人差指〜小指）


def _get_max_span(fret: int) -> int:
    """
    ポジション依存のフレットスパン上限。
    ローポジション: フレット間隔が広い → スパン3
    ミドル: スパン4
    ハイポジション: フレット間隔が狭い → スパン5
    """
    if fret <= 3:
        return 3
    elif fret <= 9:
        return 4
    else:
        return 5


def _get_position_center(fret: int) -> int:
    """フレット番号からポジション中心を返す。"""
    if fret == 0:
        return 0  # 開放弦はポジション0
    return max(1, fret - 1)  # 人差指がfret-1にある想定


def _position_cost(s: int, f: int, pitch: int = None) -> float:
    """
    位置コスト: フレットの位置自体の弾きやすさ。
    高フレットほどコストが高い。sweet spot (0-7f) は良いスコア。
    人間運指選好マップが利用可能な場合、人間が好むポジションにボーナスを付与。
    """
    cost = 0.0

    # フレット高コスト
    cost += f * WEIGHTS["w_fret_height"]

    # 9フレット超の追加コスト（ソロギターでは7-9fも多用される）
    if f > 9:
        extra = (f - 9) * WEIGHTS["w_high_fret_extra"]
        # 低弦(4-6弦)のハイフレットはさらにコスト増
        if s >= 4:
            extra *= WEIGHTS["w_low_string_high_fret"]
        cost += extra

    # Sweet spot ボーナス (負のコスト) — 0-9fまで拡大
    if 0 <= f <= 9:
        cost += WEIGHTS["w_sweet_spot_bonus"]

    # 人間運指選好ボーナス (IDMT human fingering data)
    if pitch is not None:
        cost += _human_preference_cost(pitch, s, f)

    return cost


def _transition_cost(s: int, f: int,
                     prev_s: int, prev_f: int) -> float:
    """
    遷移コスト: 前のポジションから今のポジションへの移動コスト。
    ポジション移動量 + 弦切り替え距離に基づく。
    """
    cost = 0.0

    # フレット移動コスト (開放弦の軽減を抑制してポジション連続性を重視)
    if f == 0:
        # 開放弦への移動 = 指を離すだけだが、ポジション連続性は崩れる
        fret_diff = abs(prev_f)
        cost += fret_diff * WEIGHTS["w_movement"] * 0.7
    elif prev_f == 0:
        # 開放弦からの移動 = 新しくポジションを取る
        cost += f * WEIGHTS["w_movement"] * 0.8
    else:
        # 押弦同士の移動
        fret_diff = abs(f - prev_f)
        cost += fret_diff * WEIGHTS["w_movement"]

        # ポジション跨ぎペナルティ (4フレット超のジャンプ)
        if fret_diff > POSITION_WIDTH:
            cost += (fret_diff - POSITION_WIDTH) * WEIGHTS["w_position_shift"]

    # 弦切り替えコスト
    string_dist = abs(s - prev_s)
    if string_dist > 0:
        cost += string_dist * WEIGHTS["w_string_switch"]
    else:
        # ⑧ 右手PIMA制約: 同じ弦の連打は右手の同指連打になり困難
        cost += WEIGHTS["w_same_string_repeat"]

    return cost


def _timbre_cost(s: int, f: int, tuning: List[int]) -> float:
    """
    音色コスト: 開放弦や音色的に好ましいポジションを優遇。
    開放弦のボーナスは控えめにし、過剰優遇を防ぐ。
    """
    cost = 0.0

    if f == 0:
        # 開放弦ボーナス（控えめ）
        cost += WEIGHTS["w_open_string_bonus"]

        # 開放弦でしか出せないピッチの場合に追加ボーナス
        # （他弦で同じピッチが弾けるなら開放弦にこだわる必要なし）
        string_idx = 6 - s
        if 0 <= string_idx < len(tuning):
            pitch = tuning[string_idx]
            # このピッチが他の弦でも出せるかチェック
            alt_count = 0
            for i, op in enumerate(tuning):
                if i == string_idx:
                    continue
                alt_fret = pitch - op
                if 0 < alt_fret <= MAX_FRET:  # fret>0 で弾ける代替
                    alt_count += 1
            if alt_count == 0:
                # この弦の開放弦でしか出せない音 → ボーナス
                cost += WEIGHTS["w_open_match_bonus"]

    return cost


def _ergonomic_cost_chord(combo: Tuple[Tuple[int, int], ...]) -> float:
    """
    人間工学コスト (和音用): 指の物理的制約を評価。
    - フレットスパン制約 (ポジション依存)
    - バレー判定
    - 指がかち合う判定
    """
    cost = 0.0
    all_frets = [f for _, f in combo]
    frets_nonzero = [f for f in all_frets if f > 0]

    if not frets_nonzero:
        return 0.0  # 全て開放弦 → コストゼロ

    min_fret = min(frets_nonzero)
    max_fret = max(frets_nonzero)
    span = max_fret - min_fret

    # ポジション依存のスパン制約
    max_span = _get_max_span(min_fret)
    if span > max_span:
        return WEIGHTS["w_unplayable"]  # 物理的に弾けない

    # フレットスパンコスト (スパンが広いほどコスト増)
    cost += span * WEIGHTS["w_fret_span"]

    # バレーコードボーナス: 同一フレットの複数弦を1本指で押さえる
    fret_counts = Counter(frets_nonzero)
    for fret_val, count in fret_counts.items():
        if count > 1:
            # バレーの場合、追加の弦はコストゼロに近い
            cost += (count - 1) * WEIGHTS["w_barre_bonus"]

    # 指がかち合うチェック: 低弦が高フレット・高弦が低フレットは不自然
    sorted_by_string = sorted(combo, key=lambda x: x[0], reverse=True)  # 6弦→1弦
    for i in range(len(sorted_by_string) - 1):
        s1, f1 = sorted_by_string[i]      # 低い弦
        s2, f2 = sorted_by_string[i + 1]  # 高い弦
        if f1 > 0 and f2 > 0 and f1 > f2 + 2:
            # 低弦が高弦より3フレット以上高い → 指が交差
            cost += 50.0

    # ⑨ 隣接弦ストレッチ制約 (JKU Inhibition Loss)
    # 隣接する弦で3フレット以上離れた同時押弦はストレッチが必要
    for i in range(len(sorted_by_string) - 1):
        s1, f1 = sorted_by_string[i]
        s2, f2 = sorted_by_string[i + 1]
        if f1 > 0 and f2 > 0 and abs(s1 - s2) == 1:
            fret_gap = abs(f1 - f2)
            if fret_gap > 3:
                cost += (fret_gap - 3) * WEIGHTS["w_adjacent_stretch"]

    # ⑨ 4音超の同時押弦制約 (バレーなしの場合)
    # 4本指しかないので、バレーコード以外では4弦超の同時押弦は物理的に困難
    if len(frets_nonzero) > 4:
        # バレーで同一フレットの弦は1指で押さえられるので差し引く
        unique_fret_positions = len(set(frets_nonzero))
        if unique_fret_positions > 4:
            cost += WEIGHTS["w_too_many_fingers"]

    return cost


# =============================================================================
# Viterbi DP (ISMIR "Minimax Approach" / tuttut HMM ベース)
# =============================================================================

def _viterbi_single_notes(groups: List[List[dict]], tuning: List[int],
                          max_fret: int, initial_position: float) -> List[dict]:
    """
    Viterbi DPでフレーズ内の単音列の最適パスを探索する。
    和音グループはそのまま通過させ、単音のみDPで最適化する。

    State = (string, fret) のポジション
    Observation = MIDIピッチ
    Transition cost = _transition_cost()
    Emission cost = _position_cost() + _timbre_cost()
    """
    if not groups:
        return []

    # フレーズ内の全グループを処理
    # 和音グループは事前に割り当てておく
    n_groups = len(groups)

    # 各グループの候補を列挙
    group_candidates = []  # [group_idx] -> [(string, fret), ...] or None (和音)
    chord_results = {}     # group_idx -> [assigned_notes]

    for gi, group in enumerate(groups):
        if len(group) == 1:
            note = group[0]
            positions = get_possible_positions(note["pitch"], tuning, max_fret)
            if not positions:
                # 音域外: フォールバック割り当て
                positions = [_fallback_position(note["pitch"], tuning, max_fret)]
            
            # CNN弦分類器による候補プルーニング
            cnn_probs = note.get('cnn_string_probs')
            if cnn_probs and len(positions) > 1:
                max_prob = max(cnn_probs.values())
                if max_prob > 0.5:  # CNNが自信を持っている場合のみ
                    pruned = [(s, f) for s, f in positions
                              if s in cnn_probs and cnn_probs[s] >= 0.05]
                    if len(pruned) >= 1:
                        positions = pruned
            
            group_candidates.append(positions)
        else:
            # 和音: 事前に割り当て、候補はその結果のみ
            group_candidates.append(None)

    # --- 和音を先に処理 (前後の文脈は後で調整) ---
    for gi, group in enumerate(groups):
        if len(group) > 1:
            # 前のフィンガリングを探す
            prev_f = None
            for pgi in range(gi - 1, -1, -1):
                if pgi in chord_results:
                    prev_f = [(n["string"], n["fret"]) for n in chord_results[pgi]]
                    break
                elif group_candidates[pgi] is not None:
                    break  # まだ未決定の単音
            chord_notes = [dict(n) for n in group]
            # 音楽理論: コード名を取得して典型フォーム一致コストに使用
            chord_name = group[0].get("_chord_name", "")
            assigned = _assign_chord_notes(chord_notes, tuning, max_fret, prev_f,
                                           chord_name=chord_name)
            chord_results[gi] = assigned
            # 和音の結果をViterbi用の「固定候補」として設定
            fingering = tuple((n["string"], n["fret"]) for n in assigned)
            group_candidates[gi] = [fingering[0]] if fingering else [(1, 0)]

    # --- Viterbi DP (単音のみ) ---
    # trellis[gi] = {(s, f): (cumulative_cost, backpointer)}
    trellis = [{} for _ in range(n_groups)]

    # 初期化 (最初のグループ)
    first_candidates = group_candidates[0]
    if first_candidates is None:
        first_candidates = [(1, 0)]

    for s, f in first_candidates:
        first_pitch = groups[0][0].get('pitch') if len(groups[0]) == 1 else None
        pos_cost = _position_cost(s, f, pitch=first_pitch)
        tmb_cost = _timbre_cost(s, f, tuning)
        # CNN弦推定ヒントを取り込む
        cnn_bonus = 0.0
        if len(groups[0]) == 1:
            cnn_probs = groups[0][0].get('cnn_string_probs')
            if cnn_probs and s in cnn_probs:
                cnn_bonus = -cnn_probs[s] * 30.0  # ボーナス (コスト減)
        # 初期ポジションからの距離
        init_cost = abs(f - initial_position) * WEIGHTS["w_movement"] * 0.3 if f > 0 else 0.0
        # ⑪ ソロギター用: コードフォーム内ポジション優先
        chord_form_cost = 0.0
        if len(groups[0]) == 1:
            chord_name = groups[0][0].get("_chord_name", "")
            if chord_name:
                chord_form_cost = _chord_form_position_cost(s, f, chord_name, tuning)
        total = pos_cost + tmb_cost + cnn_bonus + init_cost + chord_form_cost
        trellis[0][(s, f)] = (total, None)

    # Forward pass
    for gi in range(1, n_groups):
        candidates = group_candidates[gi]
        if candidates is None:
            candidates = [(1, 0)]

        is_single = len(groups[gi]) == 1
        prev_trellis = trellis[gi - 1]

        if not prev_trellis:
            # 前のtrellisが空の場合、初期化と同様に処理
            for s, f in candidates:
                note_pitch = groups[gi][0].get('pitch') if len(groups[gi]) == 1 else None
                pos_cost = _position_cost(s, f, pitch=note_pitch)
                tmb_cost = _timbre_cost(s, f, tuning)
                trellis[gi][(s, f)] = (pos_cost + tmb_cost, None)
            continue

        for s, f in candidates:
            best_cost = float('inf')
            best_prev = None

            # Emission cost (このポジション自体のコスト)
            note_pitch = groups[gi][0].get('pitch') if is_single else None
            pos_cost = _position_cost(s, f, pitch=note_pitch)
            tmb_cost = _timbre_cost(s, f, tuning)
            emission = pos_cost + tmb_cost

            # CNN弦分類器ヒント
            if is_single:
                cnn_probs = groups[gi][0].get('cnn_string_probs')
                if cnn_probs and s in cnn_probs:
                    emission -= cnn_probs[s] * 30.0
                # ⑪ ソロギター用: コードフォーム内ポジション優先
                chord_name = groups[gi][0].get("_chord_name", "")
                if chord_name:
                    emission += _chord_form_position_cost(s, f, chord_name, tuning)

            # 全ての前状態からの遷移を評価
            # IOI制約 (Bontempi 2024): 音符間の時間差に応じたフレット移動制限
            ioi = 999.0  # デフォルト: 制限なし
            prev_time = groups[gi - 1][0].get("start", 0)
            cur_time = groups[gi][0].get("start", 0)
            ioi = max(0.01, cur_time - prev_time)
            # 人間の指の移動速度: 約12フレット/秒が限界
            # IOI=0.1s → max_reach=1.2f, IOI=0.5s → max_reach=6f, IOI=1s → 12f
            max_fret_reach = min(MAX_FRET, max(2, int(ioi * 12)))

            for (prev_s, prev_f), (prev_cost, _) in prev_trellis.items():
                trans = _transition_cost(s, f, prev_s, prev_f)
                
                # IOI制約: 物理的に不可能なフレットジャンプにペナルティ
                fret_jump = abs(f - prev_f) if (f > 0 and prev_f > 0) else 0
                if fret_jump > max_fret_reach:
                    trans += (fret_jump - max_fret_reach) * 15.0  # 超過分に大きなペナルティ
                
                total = prev_cost + emission + trans

                if total < best_cost:
                    best_cost = total
                    best_prev = (prev_s, prev_f)

            trellis[gi][(s, f)] = (best_cost, best_prev)

    # Backtrack: 最適パスを復元
    result = []

    # 最終グループの最小コスト状態を見つける
    if not trellis[-1]:
        # フォールバック
        for group in groups:
            result.extend(group)
        return result

    best_final = min(trellis[-1].items(), key=lambda x: x[1][0])
    best_state = best_final[0]

    # バックトラック
    path = [None] * n_groups
    path[-1] = best_state

    for gi in range(n_groups - 2, -1, -1):
        next_state = path[gi + 1]
        if next_state in trellis[gi + 1]:
            _, backptr = trellis[gi + 1][next_state]
            path[gi] = backptr
        else:
            # フォールバック: trellisの最小コスト状態を使用
            if trellis[gi]:
                path[gi] = min(trellis[gi].items(), key=lambda x: x[1][0])[0]
            else:
                path[gi] = (1, 0)

    # パスの結果をノートに適用
    for gi, group in enumerate(groups):
        if gi in chord_results:
            # 和音はすでに割り当て済み
            result.extend(chord_results[gi])
        elif len(group) == 1:
            note = group[0]
            state = path[gi]
            if state:
                note["string"] = state[0]
                note["fret"] = state[1]
            else:
                # フォールバック
                positions = get_possible_positions(note["pitch"], tuning, max_fret)
                if positions:
                    note["string"] = positions[0][0]
                    note["fret"] = positions[0][1]
                else:
                    fb = _fallback_position(note["pitch"], tuning, max_fret)
                    note["string"] = fb[0]
                    note["fret"] = fb[1]
            result.append(note)
        else:
            result.extend(group)

    return result


def _minimax_postprocess(notes: List[dict], tuning: List[int],
                         max_fret: int) -> List[dict]:
    """
    Minimax Viterbi DP (Hori & Sagayama, ISMIR 2016) による後処理。
    
    通常のViterbi (sum-optimal) の結果に対して、
    Minimax基準 (最大単ステップコストの最小化) で再最適化する。
    
    更新式:
      δ_t(j) = min_i [ max(δ_{t-1}(i), step_cost(i→j)) ]
    
    これにより「1箇所だけ超難しいジャンプ」を全体的に回避する。
    sum-optimalの結果とminimax-optimalの結果を比較し、
    minimax解の方が最大ステップコストが小さい場合にのみ置換する。
    """
    if len(notes) < 3:
        return notes
    
    # --- Phase 1: 各ノートの候補ポジションを列挙 ---
    candidates_list = []
    for note in notes:
        pitch = note.get("pitch", 0)
        positions = get_possible_positions(pitch, tuning, max_fret)
        if not positions:
            positions = [(note.get("string", 1), note.get("fret", 0))]
        candidates_list.append(positions)
    
    n = len(notes)
    
    # --- Phase 2: Minimax Viterbi DP ---
    # mm_trellis[i][(s,f)] = (max_step_cost_on_path, backpointer)
    mm_trellis = [{} for _ in range(n)]
    
    # 初期化: 最初のノートのステップコストは位置コストのみ
    for s, f in candidates_list[0]:
        step_cost = _position_cost(s, f) + _timbre_cost(s, f, tuning)
        mm_trellis[0][(s, f)] = (step_cost, None)
    
    # Forward pass (minimax semiring)
    for i in range(1, n):
        # IOI計算
        prev_time = notes[i - 1].get("start", 0)
        cur_time = notes[i].get("start", 0)
        ioi = max(0.01, cur_time - prev_time)
        max_fret_reach = min(max_fret, max(2, int(ioi * 12)))
        
        for s, f in candidates_list[i]:
            best_max_cost = float('inf')
            best_prev = None
            
            emission = _position_cost(s, f) + _timbre_cost(s, f, tuning)
            
            for (prev_s, prev_f), (prev_max, _) in mm_trellis[i - 1].items():
                trans = _transition_cost(s, f, prev_s, prev_f)
                
                # IOI制約
                fret_jump = abs(f - prev_f) if (f > 0 and prev_f > 0) else 0
                if fret_jump > max_fret_reach:
                    trans += (fret_jump - max_fret_reach) * 15.0
                
                step_cost = emission + trans
                # Minimax: パス上の最大ステップコストを追跡
                path_max = max(prev_max, step_cost)
                
                if path_max < best_max_cost:
                    best_max_cost = path_max
                    best_prev = (prev_s, prev_f)
            
            if best_prev is not None:
                mm_trellis[i][(s, f)] = (best_max_cost, best_prev)
    
    if not mm_trellis[-1]:
        return notes
    
    # --- Phase 3: Backtrack minimax最適パス ---
    best_final = min(mm_trellis[-1].items(), key=lambda x: x[1][0])
    mm_path = [None] * n
    current = best_final[0]
    mm_path[-1] = current
    
    for i in range(n - 1, 0, -1):
        _, prev = mm_trellis[i][current]
        if prev is None:
            break
        mm_path[i - 1] = prev
        current = prev
    
    # --- Phase 4: sum-optimal vs minimax-optimal の比較 ---
    # 現在のパス(sum-optimal)の最大ステップコストを計算
    sum_max_step = 0.0
    for i in range(1, n):
        s, f = notes[i].get("string", 1), notes[i].get("fret", 0)
        ps, pf = notes[i-1].get("string", 1), notes[i-1].get("fret", 0)
        step = _transition_cost(s, f, ps, pf) + _position_cost(s, f)
        sum_max_step = max(sum_max_step, step)
    
    # minimax-optimalの最大ステップコスト
    mm_max_step = best_final[1][0]
    
    # minimaxの方が最大ステップコストが小さい場合にのみ置換
    # 保守的閾値: sum-optimalの最大ステップが100以上(明らかに弾けない)
    # かつminimax解が50%以上改善する場合にのみ適用
    if sum_max_step > 100.0 and mm_max_step < sum_max_step * 0.5:
        replaced = 0
        for i in range(n):
            if mm_path[i] is None:
                continue
            new_s, new_f = mm_path[i]
            old_s = notes[i].get("string", 1)
            old_f = notes[i].get("fret", 0)
            if new_s != old_s or new_f != old_f:
                notes[i]["string"] = new_s
                notes[i]["fret"] = new_f
                replaced += 1
        if replaced > 0:
            print(f"[Minimax Viterbi] {replaced}ノート改善 "
                  f"(max_step: {sum_max_step:.1f} → {mm_max_step:.1f})")
    
    return notes


def _fallback_position(pitch: int, tuning: List[int],
                       max_fret: int) -> Tuple[int, int]:
    """音域外のノートに対するフォールバックポジション。"""
    if pitch < tuning[0]:
        return (6, 0)
    elif pitch > tuning[-1] + max_fret:
        return (1, max_fret)
    else:
        best_string = 6
        min_diff = abs(pitch - tuning[0])
        for i, op in enumerate(tuning):
            diff = abs(pitch - op)
            if diff < min_diff:
                min_diff = diff
                best_string = 6 - i
        fret = max(0, min(pitch - tuning[6 - best_string], max_fret))
        return (best_string, fret)


# =============================================================================
# 和音処理
# =============================================================================

def _score_chord(combo: Tuple[Tuple[int, int], ...],
                 prev_fingering: Optional[List[Tuple[int, int]]],
                 tuning: List[int],
                 chord_name: str = "") -> float:
    """
    和音のスコアリング (高いほど良い)。
    音楽理論コスト（典型フォーム一致、ルート音制約、構成音一致）を統合。

    Parameters
    ----------
    combo : ((string, fret), ...) フィンガリング候補
    prev_fingering : 前のフィンガリング
    tuning : チューニング
    """
    # コストを負のスコアに変換
    score = 0.0

    # 1. 人間工学コスト
    ergo = _ergonomic_cost_chord(combo)
    if ergo >= WEIGHTS["w_unplayable"]:
        return -10000
    score -= ergo

    # 2. 位置コスト (平均)
    for s, f in combo:
        score -= _position_cost(s, f)

    # 3. 音色コスト
    for s, f in combo:
        score -= _timbre_cost(s, f, tuning)

    # 4. 遷移コスト
    if prev_fingering and combo:
        # 平均ポジション間の遷移
        all_prev_frets = [pf for _, pf in prev_fingering]
        avg_prev = sum(all_prev_frets) / len(all_prev_frets) if all_prev_frets else 0
        all_frets = [f for _, f in combo]
        avg_cur = sum(all_frets) / len(all_frets)

        # 代表点間の遷移コスト
        prev_s_avg = sum(ps for ps, _ in prev_fingering) / len(prev_fingering)
        cur_s_avg = sum(s for s, _ in combo) / len(combo)
        score -= _transition_cost(
            int(round(cur_s_avg)), int(round(avg_cur)),
            int(round(prev_s_avg)), int(round(avg_prev))
        )

    # ⑦ フィンガースタイル弦域分離 (SMC Fingerstyle論文)
    # 和音内のベース音(最低ピッチ)は低弦、メロディ(最高ピッチ)は高弦を優先
    if len(combo) >= 2:
        # ピッチ情報が必要なので、comboから弦→ピッチを推定
        strings = [s for s, _ in combo]
        # 最も低い弦(大きい弦番号)がベース、最も高い弦(小さい弦番号)がメロディ
        bass_string = max(strings)  # 弦番号が大きい方 = 低い弦
        melody_string = min(strings)

        # ベース音が低弦(4-6弦)にいればボーナス
        if bass_string >= 4:
            score -= WEIGHTS["w_bass_low_string"]  # 負の重み → スコア加算
        else:
            score -= WEIGHTS["w_bass_wrong_string"]  # 正の重み → スコア減算

        # メロディ音が高弦(1-3弦)にいればボーナス
        if melody_string <= 3:
            score -= WEIGHTS["w_melody_high_string"]  # 負の重み → スコア加算

    # ⑩ 音楽理論コスト (坂井論文準拠: 典型フォーム一致 + ルート音 + 構成音)
    if chord_name:
        theory_cost = _music_theory_output_cost(combo, chord_name, tuning)
        score -= theory_cost  # コストを減算 → スコア化

    return score


def _assign_chord_notes(notes: List[dict], tuning: List[int],
                        max_fret: int,
                        prev_fingering: Optional[List[Tuple[int, int]]],
                        chord_name: str = "") -> List[dict]:
    """
    和音のフィンガリング割り当て。
    全組み合わせを列挙し、_score_chord でスコアリング。
    音楽理論コスト（典型フォーム一致、ルート音制約）を統合。
    """
    # 各ノートのポジション候補を取得
    note_positions = []
    for note in notes:
        positions = get_possible_positions(note["pitch"], tuning, max_fret)
        if not positions:
            fallback_fret = min(max(0, note["pitch"] - tuning[-1]), max_fret)
            positions = [(1, fallback_fret)]
        note_positions.append(positions)

    # 組み合わせが多すぎる場合は各ノートの候補を制限
    total_combos = 1
    for p in note_positions:
        total_combos *= len(p)
    if total_combos > 5000:
        # 候補を3つまでに制限 (低フレット順)
        note_positions = [sorted(p, key=lambda x: x[1])[:3] for p in note_positions]

    best_combo = None
    best_score = -float('inf')

    for combo in iter_product(*note_positions):
        # 同じ弦に2つのノートは弾けない
        strings_used = [s for s, f in combo]
        if len(set(strings_used)) != len(strings_used):
            continue

        score = _score_chord(combo, prev_fingering, tuning, chord_name=chord_name)

        if score > best_score:
            best_score = score
            best_combo = combo

    if best_combo:
        for i, note in enumerate(notes):
            note["string"] = best_combo[i][0]
            note["fret"] = best_combo[i][1]
    else:
        # フォールバック: greedy割り当て
        used_strings = set()
        for i, note in enumerate(notes):
            for s, f in note_positions[i]:
                if s not in used_strings:
                    note["string"] = s
                    note["fret"] = f
                    used_strings.add(s)
                    break
            else:
                note["string"] = note_positions[i][0][0]
                note["fret"] = note_positions[i][0][1]

    return notes


# =============================================================================
# グルーピング
# =============================================================================

def _group_simultaneous(notes: List[dict], threshold: float = 0.03) -> List[List[dict]]:
    """Group notes that start within `threshold` seconds of each other."""
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))
    groups = [[sorted_notes[0]]]
    for n in sorted_notes[1:]:
        if abs(n["start"] - groups[-1][0]["start"]) < threshold:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


# =============================================================================
# メインの弦割り当て関数
# =============================================================================

def assign_strings_dp(notes: List[dict], tuning: List[int] = None,
                      max_fret: int = MAX_FRET,
                      initial_position: float = 0.0,
                      chords: List[dict] = None,
                      audio_path: str = None) -> List[dict]:
    """
    Assign (string, fret) to each note using Viterbi DP + Minimax postprocessing.

    研究論文ベースの改善版 + 音楽理論統合:
    - Viterbi DP: フレーズ全体で最適パス探索
    - 多属性コスト関数: 位置/遷移/人間工学/音色の4カテゴリ
    - 音楽理論コスト: 典型フォーム一致/ルート音/構成音 (坂井論文準拠)
    - Minimax後処理: 最大難度の1手を回避
    - ポジション依存フレットスパン: ローポジションはスパン3、ハイはスパン5

    Parameters
    ----------
    initial_position : float
        キー検出結果から推定される初期ポジション中心フレット。
    chords : List[dict], optional
        コード検出結果。各要素: {'start': float, 'end': float, 'chord': str}
        Viterbiの出力コストに音楽理論的制約を加えるために使用。
    """
    if tuning is None:
        tuning = STANDARD_TUNING

    if not notes:
        return notes

    # 音楽理論エンジン: 典型フォームDBをロード
    if chords:
        _load_chord_forms_db()
        print(f"[string_assigner] 音楽理論エンジン起動: {len(chords)}コード区間, "
              f"典型フォームDB={len(_CHORD_FORMS_DB or [])}個")

    # 弦分類器CNN: audio_pathが指定されている場合、各ノートに弦確率を注入
    if audio_path and os.path.exists(audio_path):
        clf = _load_string_classifier()
        if clf:
            injected = 0
            for note in notes:
                if 'cnn_string_probs' not in note:  # 既存のpropsを上書きしない
                    probs = _predict_string_probs(
                        audio_path, note.get('start', 0), note.get('pitch', 60)
                    )
                    if probs:
                        note['cnn_string_probs'] = probs
                        injected += 1
            if injected > 0:
                print(f"[string_assigner] 弦分類器CNN: {injected}/{len(notes)}ノートに弦確率注入")

    # Group simultaneous notes (within 10ms — アルペジオの順次音を分離)
    # CNN弦分類器が利用可能な場合: CNN-first + DP後段修正
    has_cnn = any(note.get('cnn_string_probs') for note in notes)
    
    if has_cnn:
        # === CNN-first アーキテクチャ + 開放弦優先 ===
        # LSTM hybrid は GuitarSet Hex→マイク音声のドメインギャップで汎化失敗のため無効化
        OPEN_STRING_PROB_THRESHOLD = 0.01
        
        for note in notes:
            cnn_probs = note.get('cnn_string_probs')
            positions = get_possible_positions(note["pitch"], tuning, max_fret)
            if not positions:
                positions = [_fallback_position(note["pitch"], tuning, max_fret)]
            
            if cnn_probs:
                # 開放弦優先チェック
                open_positions = [(s, f) for s, f in positions if f == 0]
                
                if open_positions:
                    for os_s, os_f in open_positions:
                        os_prob = cnn_probs.get(str(os_s), cnn_probs.get(os_s, 0))
                        if os_prob >= OPEN_STRING_PROB_THRESHOLD:
                            note["string"] = os_s
                            note["fret"] = os_f
                            break
                    else:
                        open_positions = None
                
                if not open_positions:
                    sorted_strings = sorted(cnn_probs.items(), key=lambda x: -x[1])
                    assigned = False
                    for s_cand, prob in sorted_strings:
                        pos_for_string = [(s, f) for s, f in positions if s == s_cand]
                        if pos_for_string:
                            note["string"] = pos_for_string[0][0]
                            note["fret"] = pos_for_string[0][1]
                            assigned = True
                            break
                    if not assigned:
                        note["string"] = positions[0][0]
                        note["fret"] = positions[0][1]
            else:
                note["string"] = positions[0][0]
                note["fret"] = positions[0][1]
        
        return notes
    
    # === 従来のViterbi DP アーキテクチャ（CNN未使用時）===
    groups = _group_simultaneous(notes, threshold=0.01)

    # フレーズに分割 (0.5秒以上の休符で区切る)
    phrases = []
    current_phrase = [groups[0]]
    for gi in range(1, len(groups)):
        prev_time = current_phrase[-1][0].get("start", 0)
        cur_time = groups[gi][0].get("start", 0)
        if cur_time - prev_time > 0.5:
            phrases.append(current_phrase)
            current_phrase = []
        current_phrase.append(groups[gi])
    if current_phrase:
        phrases.append(current_phrase)

    result = []

    for phrase in phrases:
        # 和音グループにコード情報を付与
        if chords:
            for group in phrase:
                group_time = group[0].get("start", 0)
                chord_info = _get_chord_at_time(chords, group_time)
                if chord_info:
                    for note in group:
                        note["_chord_name"] = chord_info.get("chord", "")

        # Viterbi DPでフレーズ全体を最適化
        phrase_result = _viterbi_single_notes(
            phrase, tuning, max_fret, initial_position
        )
        result.extend(phrase_result)

    # Minimax後処理: 最大遷移コストの箇所を局所再最適化
    result = _minimax_postprocess(result, tuning, max_fret)

    return result
