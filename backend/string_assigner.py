"""
string_assigner.py — 弦・フレット最適割り当て (Viterbi DP + Minimax)
=====================================================================
MIDIノート番号を (弦, フレット) に変換する。

コスト関数 → guitar_cost_functions.py
音楽理論   → chord_theory.py
チューニング定数 → solotab_utils.py
"""

from typing import List, Tuple, Optional, Dict
from itertools import product as iter_product
import os
import numpy as np

# チューニング定数 (正規定義は solotab_utils.py)
from solotab_utils import STANDARD_TUNING, TUNINGS

# コスト関数
from guitar_cost_functions import (
    MAX_FRET, WEIGHTS, POSITION_WIDTH,
    _position_cost, _transition_cost, _timbre_cost,
    _ergonomic_cost_chord, _get_max_span,
    get_finger_candidates, _bio_finger_transition_cost,
    pima_r5_postprocess,
)

# 音楽理論
from chord_theory import (
    _load_chord_forms_db, _CHORD_FORMS_DB, _CHORD_FORMS_LOOKUP,
    _parse_chord_name, _get_chord_notes_pc, _get_chord_at_time,
    _typical_form_match_cost, _music_theory_output_cost,
    _chord_form_position_cost,
)

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
# V3 FingeringTransformer — 記号ベース運指予測 (GProTab 800万ノートで学習)
# =============================================================================

_FINGERING_TRANSFORMER = None  # lazy-loaded
_FINGERING_TRANSFORMER_CTX_LEN = 16  # 文脈窓サイズ


def _load_fingering_transformer():
    """V3 FingeringTransformerモデルをlazy-load。"""
    global _FINGERING_TRANSFORMER
    if _FINGERING_TRANSFORMER is not None:
        return _FINGERING_TRANSFORMER

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..", "gp_training_data", "v3", "models",
        "fingering_transformer_v3_best.pt"
    )
    model_path = os.path.normpath(model_path)
    if not os.path.exists(model_path):
        print(f"[string_assigner] V3 Transformerモデルなし: {model_path}")
        _FINGERING_TRANSFORMER = False
        return False

    try:
        import torch
        import sys
        train_dir = os.path.join(os.path.dirname(__file__), "train")
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)
        from fingering_model_v3 import FingeringTransformer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 学習時のハイパーパラメータ: d_model=192, nhead=6, num_layers=4, embed_dim=48
        model = FingeringTransformer(
            d_model=192, nhead=6, num_layers=4, embed_dim=48, dropout=0.1
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # チェックポイント形式の場合、model_state_dictキーを抽出
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval()

        _FINGERING_TRANSFORMER = {'model': model, 'device': device}
        print(f"[string_assigner] V3 FingeringTransformerロード完了 (device={device})")
        return _FINGERING_TRANSFORMER
    except Exception as e:
        print(f"[string_assigner] V3 Transformerロード失敗: {e}")
        _FINGERING_TRANSFORMER = False
        return False


def _predict_string_transformer(notes: List[dict], tuning: List[int] = None) -> List[dict]:
    """
    V3 FingeringTransformerで弦予測確率を各ノートに注入する。

    過去の弦・フレット割り当て結果を文脈として使用し、
    各ノートに 'transformer_string_probs' フィールドを追加する。

    Parameters
    ----------
    notes : 弦・フレットが既に割り当て済みのノートリスト
    tuning : チューニング

    Returns
    -------
    notes : transformer_string_probs が注入されたノートリスト
    """
    transformer = _load_fingering_transformer()
    if not transformer:
        return notes

    if tuning is None:
        tuning = STANDARD_TUNING

    import torch

    model = transformer['model']
    device = transformer['device']
    ctx_len = _FINGERING_TRANSFORMER_CTX_LEN
    injected = 0

    for idx, note in enumerate(notes):
        pitch = note.get('pitch', 60)

        # 文脈窓: 直前ctx_len個のノート（割り当て済み弦・フレットを使用）
        ctx_pitches = []
        ctx_strings = []
        ctx_frets = []
        ctx_durations = []
        ctx_intervals = []

        start = max(0, idx - ctx_len)
        context_notes = notes[start:idx]

        prev_pitch = pitch  # 逆順に辿る用
        for ci, cn in enumerate(context_notes):
            cp = cn.get('pitch', 60)
            cs = cn.get('string', 0)
            cf = cn.get('fret', 0)
            # duration: 簡易量子化 (秒→0-31)
            dur = cn.get('duration', 0.5)
            dur_q = min(31, int(dur * 8))
            # interval: 前ノートとのピッチ差 (-24~+24 → 0~48)
            if ci > 0:
                prev_p = context_notes[ci - 1].get('pitch', cp)
                interval = max(-24, min(24, cp - prev_p)) + 24
            else:
                interval = 24  # 0 (no interval)

            ctx_pitches.append(min(127, max(0, cp)))
            ctx_strings.append(min(6, max(0, cs)))
            ctx_frets.append(min(24, max(0, cf)))
            ctx_durations.append(dur_q)
            ctx_intervals.append(interval)

        # パディング (文脈が足りない場合は左側をゼロ埋め)
        pad_len = ctx_len - len(ctx_pitches)
        if pad_len > 0:
            ctx_pitches = [0] * pad_len + ctx_pitches
            ctx_strings = [0] * pad_len + ctx_strings
            ctx_frets = [0] * pad_len + ctx_frets
            ctx_durations = [0] * pad_len + ctx_durations
            ctx_intervals = [24] * pad_len + ctx_intervals

        # ターゲット特徴量
        target_dur = min(31, int(note.get('duration', 0.5) * 8))
        if idx > 0:
            target_interval = max(-24, min(24, pitch - notes[idx - 1].get('pitch', pitch))) + 24
        else:
            target_interval = 24

        # position_context: 直近8ノートのフレット中央値
        recent_frets = [n.get('fret', 0) for n in notes[max(0, idx - 8):idx] if n.get('fret', 0) > 0]
        pos_ctx = int(np.median(recent_frets)) if recent_frets else 0
        pos_ctx = min(24, max(0, pos_ctx))

        # テンソル化
        with torch.no_grad():
            t_cp = torch.LongTensor([ctx_pitches]).to(device)
            t_cs = torch.LongTensor([ctx_strings]).to(device)
            t_cf = torch.LongTensor([ctx_frets]).to(device)
            t_cd = torch.LongTensor([ctx_durations]).to(device)
            t_ci = torch.LongTensor([ctx_intervals]).to(device)
            t_tp = torch.LongTensor([min(127, max(0, pitch))]).to(device)
            t_td = torch.LongTensor([target_dur]).to(device)
            t_ti = torch.LongTensor([target_interval]).to(device)
            t_pc = torch.LongTensor([pos_ctx]).to(device)

            logits = model(t_cp, t_cs, t_cf, t_cd, t_ci, t_tp, t_td, t_ti, t_pc)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 物理的に弾けるポジションのみに制限
        positions = get_possible_positions(pitch, tuning)
        valid_strings = set(s for s, f in positions)

        # 弾けない弦の確率をゼロにして再正規化
        filtered = {}
        total = 0.0
        for s_idx in range(6):
            s_num = s_idx + 1
            if s_num in valid_strings:
                filtered[s_num] = float(probs[s_idx])
                total += float(probs[s_idx])

        if total > 0:
            for s_num in filtered:
                filtered[s_num] /= total
            note['transformer_string_probs'] = filtered
            injected += 1

    if injected > 0:
        print(f"[string_assigner] V3 Transformer: {injected}/{len(notes)}ノートに弦確率注入")

    return notes




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
                # cnn_probsのキーは文字列('1'-'6')、positionsの弦番号は整数(1-6)
                max_prob = max(cnn_probs.values())
                if max_prob > 0.5:  # CNNが自信を持っている場合のみ
                    pruned = [(s, f) for s, f in positions
                              if str(s) in cnn_probs and cnn_probs[str(s)] >= 0.05]
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

    # --- Viterbi DP (単音のみ) --- §4.2 Bio Viterbi: 状態空間 (弦, フレット, 指)
    # trellis[gi] = {(s, f, finger): (cumulative_cost, backpointer)}
    trellis = [{} for _ in range(n_groups)]

    # 初期化 (最初のグループ)
    first_candidates = group_candidates[0]
    if first_candidates is None:
        first_candidates = [(1, 0)]

    for s, f in first_candidates:
        first_pitch = groups[0][0].get('pitch') if len(groups[0]) == 1 else None
        pos_cost = _position_cost(s, f, pitch=first_pitch)
        tmb_cost = _timbre_cost(s, f, tuning)
        # --- 論文§12.2: 初期化でも3アプローチ統合 ---
        cnn_bonus = 0.0
        tfm_bonus = 0.0
        if len(groups[0]) == 1:
            cnn_probs = groups[0][0].get('cnn_string_probs')
            if cnn_probs and str(s) in cnn_probs:
                cnn_bonus = -cnn_probs[str(s)] * 12.0
            tfm_probs = groups[0][0].get('transformer_string_probs')
            if tfm_probs and str(s) in tfm_probs:
                tfm_bonus = -tfm_probs[str(s)] * 8.0
        # 初期ポジションからの距離
        init_cost = abs(f - initial_position) * WEIGHTS["w_movement"] * 0.3 if f > 0 else 0.0
        # ⑪ ソロギター用: コードフォーム内ポジション優先
        chord_form_cost = 0.0
        if len(groups[0]) == 1:
            chord_name = groups[0][0].get("_chord_name", "")
            if chord_name:
                chord_form_cost = _chord_form_position_cost(s, f, chord_name, tuning)
        total = pos_cost + tmb_cost + cnn_bonus + tfm_bonus + init_cost + chord_form_cost
        # Bio Viterbi: 各指候補で状態を展開
        for finger in get_finger_candidates(f):
            trellis[0][(s, f, finger)] = (total, None)

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
                for finger in get_finger_candidates(f):
                    trellis[gi][(s, f, finger)] = (pos_cost + tmb_cost, None)
            continue

        for s, f in candidates:
            # Emission cost (このポジション自体のコスト)
            note_pitch = groups[gi][0].get('pitch') if is_single else None
            pos_cost = _position_cost(s, f, pitch=note_pitch)
            tmb_cost = _timbre_cost(s, f, tuning)
            emission = pos_cost + tmb_cost

            # --- 論文§12.2: 3アプローチ統合 ---
            # [音声パス] CNN弦確率 → emission統合
            # [記号パス] Transformer弦確率 → emission統合（2パス目のみ有効）
            # [人間工学パス] position_cost + timbre_cost (上で計算済み)
            if is_single:
                cnn_probs = groups[gi][0].get('cnn_string_probs')
                if cnn_probs and str(s) in cnn_probs:
                    emission -= cnn_probs[str(s)] * 12.0
                tfm_probs = groups[gi][0].get('transformer_string_probs')
                if tfm_probs and str(s) in tfm_probs:
                    emission -= tfm_probs[str(s)] * 8.0
                # ⑪ ソロギター用: コードフォーム内ポジション優先
                chord_name = groups[gi][0].get("_chord_name", "")
                if chord_name:
                    emission += _chord_form_position_cost(s, f, chord_name, tuning)

            # 全ての前状態からの遷移を評価
            ioi = 999.0
            prev_time = groups[gi - 1][0].get("start", 0)
            cur_time = groups[gi][0].get("start", 0)
            ioi = max(0.01, cur_time - prev_time)
            max_fret_reach = min(MAX_FRET, max(2, int(ioi * 12)))

            # Bio Viterbi: 各指候補で状態を展開
            for finger in get_finger_candidates(f):
                best_cost = float('inf')
                best_prev = None

                for (prev_s, prev_f, prev_finger), (prev_cost, _) in prev_trellis.items():
                    # 法則3: ピッチ近接性弦保持
                    cur_pitch = groups[gi][0].get('pitch') if is_single else None
                    prev_pitch = groups[gi-1][0].get('pitch') if len(groups[gi-1]) == 1 else None
                    trans = _transition_cost(s, f, prev_s, prev_f, dt=ioi,
                                            pitch=cur_pitch, prev_pitch=prev_pitch)
                    
                    # §4.2 Bio: 生体力学的指遷移コスト
                    trans += _bio_finger_transition_cost(
                        finger, prev_finger, f, prev_f, s, prev_s)
                    
                    # IOI制約: 物理的に不可能なフレットジャンプにペナルティ
                    fret_jump = abs(f - prev_f) if (f > 0 and prev_f > 0) else 0
                    if fret_jump > max_fret_reach:
                        trans += (fret_jump - max_fret_reach) * 15.0
                    
                    total = prev_cost + emission + trans

                    if total < best_cost:
                        best_cost = total
                        best_prev = (prev_s, prev_f, prev_finger)

                trellis[gi][(s, f, finger)] = (best_cost, best_prev)

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
                path[gi] = (1, 0, 0)

    # パスの結果をノートに適用 (Bio状態 (s, f, finger) → (s, f) 抽出)
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
                note["finger"] = state[2]  # Bio指情報も保存
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
        note_pitch = notes[0].get('pitch') if notes else None
        step_cost = _position_cost(s, f, pitch=note_pitch) + _timbre_cost(s, f, tuning)
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
            
            note_pitch_i = notes[i].get('pitch') if i < len(notes) else None
            emission = _position_cost(s, f, pitch=note_pitch_i) + _timbre_cost(s, f, tuning)
            
            for (prev_s, prev_f), (prev_max, _) in mm_trellis[i - 1].items():
                trans = _transition_cost(s, f, prev_s, prev_f, dt=ioi)
                
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
        step = _transition_cost(s, f, ps, pf) + _position_cost(s, f, pitch=notes[i].get('pitch'))
        sum_max_step = max(sum_max_step, step)
    
    # minimax-optimalの最大ステップコスト
    mm_max_step = best_final[1][0]
    
    # minimaxの方が最大ステップコストが小さい場合にのみ置換
    # 保守的閾値: sum-optimalの最大ステップが100以上(明らかに弾けない)
    # かつminimax解が50%以上改善する場合にのみ適用
    if sum_max_step > 50.0 and mm_max_step < sum_max_step * 0.75:
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
    # CNN弦分類器の確率はViterbi DP内でcnn_string_probsフィールド経由で活用される。
    # CNN-firstモード（DPバイパス）は遷移コストを無視するため廃止。
    # 常にViterbi DPで全体最適化する。
    has_cnn = any(note.get('cnn_string_probs') for note in notes)
    if has_cnn:
        print(f"[string_assigner] CNN弦分類器ヒント: {sum(1 for n in notes if n.get('cnn_string_probs'))}音に適用 (Viterbi DPで統合)")
    
    # === Viterbi DP アーキテクチャ ===
    # 論文§12.2: [音声パス] CNN弦確率 + [人間工学パス] 選好マップ → emission統合
    # [記号パス] Transformer V3 は Viterbi 1パス目の結果を文脈として使用するため、
    # DP後に呼び出す（Transformerの入力に弦・フレット文脈が必要）
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

    # PIMA R5後処理: a-m-a交替パターンの回避 (Skarha 2018)
    result = pima_r5_postprocess(result, tuning, max_fret)

    # --- 論文§12.2 [記号パス]: 2パスViterbi ---
    # 1パス目: CNN + 人間工学パスのみでViterbi DP実行（上記で完了）
    # TransformerはViterbi 1パス目の弦・フレット結果を文脈として使用するため、
    # DP後に確率を注入し、2パス目のViterbi DPで全3パスを統合する。
    result = _predict_string_transformer(result, tuning)
    has_tfm = any(note.get('transformer_string_probs') for note in result)
    if has_tfm:
        tfm_count = sum(1 for n in result if n.get('transformer_string_probs'))
        print(f"[string_assigner] Transformer V3: {tfm_count}音に弦確率注入 → 2パス目Viterbi DP開始")
        # 2パス目: Transformer確率もemissionに統合してViterbi DP再実行
        groups2 = _group_simultaneous(result, threshold=0.01)
        phrases2 = []
        current_phrase2 = [groups2[0]]
        for gi in range(1, len(groups2)):
            prev_time = current_phrase2[-1][0].get("start", 0)
            cur_time = groups2[gi][0].get("start", 0)
            if cur_time - prev_time > 0.5:
                phrases2.append(current_phrase2)
                current_phrase2 = []
            current_phrase2.append(groups2[gi])
        if current_phrase2:
            phrases2.append(current_phrase2)
        
        result2 = []
        for phrase in phrases2:
            if not phrase:
                continue
            first_note = phrase[0][0]
            initial_position = first_note.get("fret", 0)
            phrase_result = _viterbi_single_notes(
                phrase, tuning, max_fret, initial_position
            )
            result2.extend(phrase_result)
        
        # 2パス目の結果で置き換え
        result = result2
        result = _minimax_postprocess(result, tuning, max_fret)
        result = pima_r5_postprocess(result, tuning, max_fret)
        print(f"[string_assigner] 2パスViterbi完了: 全3パス(CNN+Transformer+人間工学)統合")

    # コードポジション連動後処理: 無効化
    # クラシックギターではメロディがハイポジションでもアルペジオは開放弦で弾くため
    # この後処理は逆効果（82%→75%に悪化）。将来的にはジャンル判定後に適用を検討。
    # if chords:
    #     result = _chord_position_postprocess(result, chords, tuning, max_fret)

    return result
