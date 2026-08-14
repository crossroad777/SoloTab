"""
chord_detector.py — コード検出 (BTC primary + chroma fallback)
====================================================================
BTC (Bi-directional Transformer for Chords) をプライマリ検出器として使用。
BTC が利用不可の場合は librosa chroma テンプレートマッチングにフォールバック。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import librosa
from typing import List, Dict, Tuple

# BTC ラベル → SoloTab 内部形式の変換テーブル
# BTC large_voca: 'C', 'C:min', 'C:7', 'C:min7', 'N' etc.
# SoloTab:        'C', 'Cm',    'C7',  'Cm7',     'N.C.' etc.
def _btc_label_to_solotab(label: str) -> str:
    """BTC のコードラベルを SoloTab 形式に変換"""
    if label in ('N', 'X'):
        return 'N.C.'
    # 'A:min' -> 'Am', 'E:min7' -> 'Em7', 'G:maj' -> 'G'
    label = label.replace(':maj7', 'maj7')
    label = label.replace(':maj', '')
    label = label.replace(':min7', 'm7')
    label = label.replace(':min', 'm')
    label = label.replace(':7', '7')
    label = label.replace(':dim', 'dim')
    label = label.replace(':aug', 'aug')
    label = label.replace(':sus4', 'sus4')
    label = label.replace(':sus2', 'sus2')
    label = label.replace(':hdim7', 'm7b5')
    label = label.replace(':', '')  # 残りのコロンを除去
    return label


def detect_chords_btc(wav_path: str) -> List[Dict]:
    """
    BTC モデルによるコード検出。
    
    Returns
    -------
    list[dict] or None
        成功時: [{start, end, chord, confidence}], 失敗時: None
    """
    try:
        import sys, os
        # BTC エンジンのパスを追加
        btc_engine_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'nextchord', 'fastapi-backend'
        )
        # D:\Music\nextchord\fastapi-backend に btc_engine.py がある
        # ただし直接パスが異なる可能性があるので、複数パスを試す
        candidate_paths = [
            btc_engine_dir,
            r'D:\Music\nextchord\fastapi-backend',
        ]
        btc_dir = r'/app/nextchord/BTC-ISMIR19' if os.path.exists(r'/app/nextchord/BTC-ISMIR19') else r'D:\Music\nextchord\BTC-ISMIR19'
        if btc_dir not in sys.path:
            sys.path.insert(0, btc_dir)
        
        for cp in candidate_paths:
            if cp not in sys.path and os.path.isdir(cp):
                sys.path.insert(0, cp)
        
        from btc_engine import get_btc_engine
        
        engine = get_btc_engine()
        seg_starts, seg_labels = engine.detect_chords(str(wav_path))
        
        if len(seg_starts) == 0:
            return None
        
        # (starts, labels) → [{start, end, chord}] 形式に変換
        chords = []
        for i in range(len(seg_starts)):
            start = float(seg_starts[i])
            end = float(seg_starts[i + 1]) if i + 1 < len(seg_starts) else start + 2.0
            label = _btc_label_to_solotab(str(seg_labels[i]))
            chords.append({
                'start': start,
                'end': end,
                'chord': label,
                'confidence': 0.85,  # BTC は全体的に高精度
                '_source': 'btc',
            })
        
        # N.C. を前のコードで埋める
        for i in range(1, len(chords)):
            if chords[i]['chord'] == 'N.C.':
                chords[i]['chord'] = chords[i - 1]['chord']
        
        # 連続同一コードをマージ
        merged = [chords[0]] if chords else []
        for c in chords[1:]:
            if merged and merged[-1]['chord'] == c['chord']:
                merged[-1]['end'] = c['end']
            else:
                merged.append(c)
        
        print(f"[chord_detector] BTC: {len(merged)} chord regions detected")
        return merged
    
    except Exception as e:
        print(f"[chord_detector] BTC unavailable: {e}")
        return None


# コードテンプレート (ピッチクラスセット) — chroma fallback 用
CHORD_TEMPLATES = {}
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

for i, note in enumerate(NOTES):
    # メジャー (0, 4, 7)
    template = np.zeros(12)
    template[i] = 1.0
    template[(i + 4) % 12] = 0.8
    template[(i + 7) % 12] = 0.8
    CHORD_TEMPLATES[note] = template / np.linalg.norm(template)

    # マイナー (0, 3, 7)
    template_m = np.zeros(12)
    template_m[i] = 1.0
    template_m[(i + 3) % 12] = 0.8
    template_m[(i + 7) % 12] = 0.8
    CHORD_TEMPLATES[note + 'm'] = template_m / np.linalg.norm(template_m)

    # セブンス (0, 4, 7, 10)
    template_7 = np.zeros(12)
    template_7[i] = 1.0
    template_7[(i + 4) % 12] = 0.7
    template_7[(i + 7) % 12] = 0.7
    template_7[(i + 10) % 12] = 0.5
    CHORD_TEMPLATES[note + '7'] = template_7 / np.linalg.norm(template_7)


# =============================================================================
# ダイアトニックコード理論の定義とヘルパー
# =============================================================================

NOTE_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
    'A#': 10, 'Bb': 10, 'B': 11
}

def get_chord_notes_pc_by_name(chord_name: str) -> List[int]:
    """コード名から構成音のピッチクラスリストを取得"""
    from chord_theory import _parse_chord_name, _get_chord_notes_pc
    root_pc, quality = _parse_chord_name(chord_name)
    if root_pc < 0:
        return []
    return _get_chord_notes_pc(root_pc, quality)

def get_chord_diatonic_status(chord_name: str, key: str) -> str:
    """
    指定されたキーに対してコードがダイアトニック、セカンダリー（借用含む）、またはノンダイアトニックかを判定する。
    """
    if not key or not chord_name or chord_name in ('N.C.', 'N', 'X'):
        return 'non_diatonic'
        
    chord_pcs = get_chord_notes_pc_by_name(chord_name)
    if not chord_pcs:
        return 'non_diatonic'
        
    # キー名の正規化
    key_clean = key.strip()
    is_minor = key_clean.endswith('m')
    root_name = key_clean[:-1] if is_minor else key_clean
    if root_name not in NOTE_TO_PC:
        return 'non_diatonic'
        
    root_pc = NOTE_TO_PC[root_name]
    
    # 音階のピッチクラスリスト
    scales = []
    if is_minor:
        # 自然短音階: 0, 2, 3, 5, 7, 8, 10
        scales.append([(root_pc + d) % 12 for d in [0, 2, 3, 5, 7, 8, 10]])
        # 和声短音階: 0, 2, 3, 5, 7, 8, 11 (V7やvii°7をカバー)
        scales.append([(root_pc + d) % 12 for d in [0, 2, 3, 5, 7, 8, 11]])
    else:
        # 長音階: 0, 2, 4, 5, 7, 9, 11
        scales.append([(root_pc + d) % 12 for d in [0, 2, 4, 5, 7, 9, 11]])
        
    # すべての構成音が音階に含まれている場合はダイアトニック
    for scale in scales:
        if all(pc in scale for pc in chord_pcs):
            return 'diatonic'
            
    # よくあるセカンダリードミナントや借用和音の判定
    from chord_theory import _parse_chord_name
    chord_root_pc, chord_quality = _parse_chord_name(chord_name)
    rel_pc = (chord_root_pc - root_pc) % 12
    
    if is_minor:
        # マイナーキーの主な準ダイアトニック/借用和音:
        # IV (メジャーIV, メロディックマイナー由来)
        # bII (ナポリの和音, e.g. AmキーでのBb)
        # II / II7 (ダブルドミナント V/V, e.g. AmキーでのB / B7)
        if rel_pc == 2 and chord_quality in ('major', '7'):  # II / II7 (V/V)
            return 'secondary'
        if rel_pc == 5 and chord_quality in ('major', 'maj7'):  # IV
            return 'secondary'
        if rel_pc == 1 and chord_quality in ('major', 'maj7'):  # bII
            return 'secondary'
    else:
        # メジャーキーの主な準ダイアトニック/借用和音:
        # II / II7 (ダブルドミナント V/V, e.g. CキーでのD / D7)
        # III / III7 (V/vi, e.g. CキーでのE / E7)
        # VI / VI7 (V/ii, e.g. CキーでのA / A7)
        # iv (同主調マイナーからの借用, e.g. CキーでのFm)
        # bVII (同主調マイナーからの借用, e.g. CキーでのBb)
        # bVI (同主調マイナーからの借用, e.g. CキーでのAb)
        # bIII (同主調マイナーからの借用, e.g. CキーでのEb)
        if rel_pc == 2 and chord_quality in ('major', '7'):  # II / II7 (V/V)
            return 'secondary'
        if rel_pc == 4 and chord_quality in ('major', '7'):  # III / III7 (V/vi)
            return 'secondary'
        if rel_pc == 9 and chord_quality in ('major', '7'):  # VI / VI7 (V/ii)
            return 'secondary'
        if rel_pc == 5 and chord_quality in ('minor', 'm7'):  # iv
            return 'secondary'
        if rel_pc == 10 and chord_quality in ('major', 'maj7', '7'):  # bVII
            return 'secondary'
        if rel_pc == 8 and chord_quality in ('major', 'maj7'):  # bVI
            return 'secondary'
        if rel_pc == 3 and chord_quality in ('major', 'maj7'):  # bIII
            return 'secondary'
            
    return 'non_diatonic'


def get_candidate_chords_for_key(key: str, audio_chord: str) -> List[str]:
    """指定されたキーに基づいて、探索候補のコードリストを生成する"""
    candidates = set()
    if audio_chord and audio_chord != 'N.C.':
        candidates.add(audio_chord)
        
    if not key:
        return list(candidates)
        
    is_minor = key.endswith('m')
    root_name = key[:-1] if is_minor else key
    if root_name not in NOTE_TO_PC:
        return list(candidates)
        
    root_pc = NOTE_TO_PC[root_name]
    
    # ダイアトニックな度数からコードを作成
    if is_minor:
        # マイナーキー: i, ii°, III, iv, v/V, VI, VII の三和音・七和音
        degrees = [
            (0, 'm'), (0, 'm7'),  # i
            (2, 'dim'), (2, 'm7b5'),  # ii°
            (3, ''), (3, 'maj7'),  # III
            (5, 'm'), (5, 'm7'),  # iv
            (7, 'm'), (7, 'm7'), (7, ''), (7, '7'),  # v / V / V7
            (8, ''), (8, 'maj7'),  # VI
            (10, ''), (10, '7')  # VII
        ]
    else:
        # メジャーキー: I, ii, iii, IV, V, vi, vii° の三和音・七和音
        degrees = [
            (0, ''), (0, 'maj7'),  # I
            (2, 'm'), (2, 'm7'),  # ii
            (4, 'm'), (4, 'm7'),  # iii
            (5, ''), (5, 'maj7'),  # IV
            (7, ''), (7, '7'),  # V / V7
            (9, 'm'), (9, 'm7'),  # vi
            (11, 'dim'), (11, 'm7b5')  # vii°
        ]
        
    for rel_pc, suffix in degrees:
        chord_root_pc = (root_pc + rel_pc) % 12
        chord_root_name = NOTES[chord_root_pc]
        candidates.add(chord_root_name + suffix)
        
    # 主要なセカンダリードミナントや借用和音も候補に追加
    if is_minor:
        for rel_pc, suffix in [(2, '7'), (1, ''), (5, '')]:
            chord_root_name = NOTES[(root_pc + rel_pc) % 12]
            candidates.add(chord_root_name + suffix)
    else:
        for rel_pc, suffix in [(2, '7'), (4, '7'), (9, '7'), (5, 'm'), (10, ''), (8, ''), (3, '')]:
            chord_root_name = NOTES[(root_pc + rel_pc) % 12]
            candidates.add(chord_root_name + suffix)
            
    candidates.add('N.C.')
    return list(candidates)


# 度数ベースの共通進行遷移スコア定義
COMMON_PROGRESSIONS = {
    # V7 -> I, V -> I (ドミナント解決)
    (7, '7', 0, 'major'): 1.2,
    (7, '7', 0, 'maj7'): 1.2,
    (7, 'major', 0, 'major'): 1.0,
    (7, 'major', 0, 'maj7'): 1.0,
    (7, '7', 9, 'minor'): 1.2,  # V7 -> vi (偽解決、マイナーへのドミナント解決)
    (7, '7', 9, 'm7'): 1.2,
    
    # ii -> V -> I (ツーファイブワン)
    (2, 'minor', 7, '7'): 1.0,
    (2, 'minor', 7, 'major'): 0.8,
    (2, 'm7', 7, '7'): 1.0,
    
    # IV -> V (サブドミナント -> ドミナント)
    (5, 'major', 7, '7'): 0.8,
    (5, 'major', 7, 'major'): 0.8,
    (5, 'maj7', 7, '7'): 0.8,
    
    # IV -> I (サブドミナント終止)
    (5, 'major', 0, 'major'): 0.7,
    
    # I -> IV (主和音 -> サブドミナント)
    (0, 'major', 5, 'major'): 0.6,
    (0, 'maj7', 5, 'major'): 0.6,
    
    # vi -> ii (マイナー進行)
    (9, 'minor', 2, 'minor'): 0.7,
    (9, 'm7', 2, 'm7'): 0.7,
    
    # iii -> vi
    (4, 'minor', 9, 'minor'): 0.7,
    
    # マイナーキーの進行解決
    # V7 -> i (e.g. E7 -> Am in Am key)
    (7, '7', 0, 'minor'): 1.2,
    (7, '7', 0, 'm7'): 1.2,
    (7, 'major', 0, 'minor'): 1.0,
    # ii° -> V -> i
    (2, 'dim', 7, '7'): 1.0,
    (2, 'm7b5', 7, '7'): 1.0,
    # bVI -> V (ナポリ/フリジアン解決含む)
    (8, 'major', 7, '7'): 0.8,
    (8, 'major', 7, 'major'): 0.8,
}

def get_chord_degree(chord_name: str, key: str) -> Tuple[int, str]:
    """
    キーに対するコードの相対ルート半音数 (0-11) とクオリティを返す。
    """
    from chord_theory import _parse_chord_name
    root_pc, quality = _parse_chord_name(chord_name)
    if root_pc < 0:
        return (-1, '')
    
    is_minor = key.endswith('m')
    key_root = key[:-1] if is_minor else key
    key_root_pc = NOTE_TO_PC.get(key_root, 0)
    
    rel_pc = (root_pc - key_root_pc) % 12
    return (rel_pc, quality)

def refine_chords_with_notes(chords: List[Dict], notes: List[Dict], key: str) -> List[Dict]:
    """
    検出された単音（notes）とダイアトニックコード理論を用いて、オーディオベースのコード検出結果を洗練・補正する。
    """
    if not chords or not notes or not key:
        return chords
        
    refined_chords = []
    from chord_theory import _parse_chord_name
    
    for c in chords:
        start = c['start']
        end = c['end']
        audio_chord = c['chord']
        confidence = c.get('confidence', 0.8)
        
        # 1. この区間で発音している音符（notes）を収集
        segment_notes = [
            n for n in notes
            if start <= n['start'] < end or (n['start'] < start and n.get('end', n['start'] + 0.5) > start + 0.1)
        ]
        
        if not segment_notes:
            refined_chords.append(c)
            continue
            
        # ピッチクラスごとの重み付け（velocityと長さを考慮）
        pc_weights = np.zeros(12)
        lowest_pitch = 999
        lowest_pc = -1
        highest_pitch = -1
        
        for n in segment_notes:
            pitch = n.get('pitch', 60)
            pc = pitch % 12
            vel = float(n.get('velocity', 0.5))
            dur = float(n.get('end', n['start'] + 0.5)) - float(n['start'])
            weight = vel * max(0.1, dur)
            pc_weights[pc] += weight
            
            if pitch < lowest_pitch:
                lowest_pitch = pitch
                lowest_pc = pc
            if pitch > highest_pitch:
                highest_pitch = pitch
                
        total_weight = np.sum(pc_weights)
        if total_weight < 0.01:
            refined_chords.append(c)
            continue
            
        pc_probs = pc_weights / total_weight
        highest_pc = highest_pitch % 12 if highest_pitch >= 0 else -1
        
        # 2. 候補コードをスコアリング
        candidates = get_candidate_chords_for_key(key, audio_chord)
        
        best_chord = audio_chord
        best_score = -1.0
        
        prev_chord = refined_chords[-1]['chord'] if refined_chords else None
        
        for cand in candidates:
            if cand == 'N.C.':
                # 音符が鳴っている場合はN.C.のスコアを低くする
                score = 0.1
            else:
                cand_pcs = get_chord_notes_pc_by_name(cand)
                if not cand_pcs:
                    continue
                    
                # A. 単音との一致スコア (発音確率の総和)
                match_ratio = sum(pc_probs[pc] for pc in cand_pcs)
                note_score = 3.0 * match_ratio
                
                # 精度ボーナス: 候補コードの音のうち、実際に鳴っている音の割合
                num_present = sum(1 for pc in cand_pcs if pc_weights[pc] > 0)
                precision = num_present / len(cand_pcs) if cand_pcs else 0
                note_score += 0.5 * precision
                
                # A2. メロディ音（最高音）一致ボーナス
                if highest_pc >= 0 and highest_pc in cand_pcs:
                    note_score += 0.8
                
                # B. ベース音一致ボーナス (最低音がコードのルート音と一致するか)
                cand_root_pc, _ = _parse_chord_name(cand)
                if lowest_pc == cand_root_pc:
                    note_score += 1.5  # ルートがベースにある場合の大きな報酬
                elif lowest_pc in cand_pcs:
                    note_score += 0.5  # 転回形の場合の小さな報酬
                    
                # C. ダイアトニック確率（Prior）
                status = get_chord_diatonic_status(cand, key)
                if status == 'diatonic':
                    diatonic_score = 1.2
                elif status == 'secondary':
                    diatonic_score = 0.6
                else:
                    diatonic_score = 0.0
                    
                # D. オーディオ検出結果との一致スコア
                audio_score = 0.0
                if cand == audio_chord:
                    audio_score = 1.5 * confidence
                    
                # E. コード遷移（進行）ボーナス
                trans_score = 0.0
                if prev_chord:
                    prev_degree = get_chord_degree(prev_chord, key)
                    curr_degree = get_chord_degree(cand, key)
                    trans_score = COMMON_PROGRESSIONS.get(
                        (prev_degree[0], prev_degree[1], curr_degree[0], curr_degree[1]),
                        0.0
                    )
                    
                score = note_score + diatonic_score + audio_score + trans_score
                
            if score > best_score:
                best_score = score
                best_chord = cand
                
        if best_chord != audio_chord:
            print(f"[chord_refiner] Corrected segment {start:.1f}-{end:.1f}s: {audio_chord} -> {best_chord} (key: {key})")
            
        refined_chords.append({
            'start': start,
            'end': end,
            'chord': best_chord,
            'confidence': min(1.0, float(best_score / 6.0)),
            '_source': 'diatonic_refinement',
            '_original_audio_chord': audio_chord
        })
        
    if not refined_chords:
        return chords
        
    # 連続する同一コードをマージ
    merged = [refined_chords[0]]
    for c in refined_chords[1:]:
        if merged[-1]['chord'] == c['chord']:
            merged[-1]['end'] = c['end']
        else:
            merged.append(c)
            
    return merged


def detect_chords_chroma(wav_path: str, beats: List[float] = None, key: str = None,
                         sr: int = 22050, hop_length: int = 512) -> List[Dict]:
    """Chroma テンプレートマッチングによるコード検出 (フォールバック)"""
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    
    # Chroma特徴量
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    
    # ビート単位でchromaを平均化
    if beats and len(beats) >= 2:
        segments = []
        for i in range(len(beats) - 1):
            start_t = beats[i]
            end_t = beats[i + 1]
            mask = (times >= start_t) & (times < end_t)
            if np.any(mask):
                avg_chroma = np.mean(chroma[:, mask], axis=1)
            else:
                avg_chroma = np.zeros(12)
            segments.append((start_t, end_t, avg_chroma))
    else:
        interval = 0.5
        duration = len(y) / sr
        segments = []
        for t in np.arange(0, duration, interval):
            end_t = min(t + interval, duration)
            mask = (times >= t) & (times < end_t)
            if np.any(mask):
                avg_chroma = np.mean(chroma[:, mask], axis=1)
            else:
                avg_chroma = np.zeros(12)
            segments.append((t, end_t, avg_chroma))

    # 各セグメントでテンプレートマッチング
    chords = []
    for start_t, end_t, seg_chroma in segments:
        norm = np.linalg.norm(seg_chroma)
        if norm < 0.01:
            chords.append({"start": float(start_t), "end": float(end_t),
                          "chord": "N.C.", "confidence": 0.0})
            continue

        seg_norm = seg_chroma / norm
        best_chord = "N.C."
        best_score = -1.0

        for name, template in CHORD_TEMPLATES.items():
            score = float(np.dot(seg_norm, template))
            
            # ダイアトニック優先バイアスの適用
            if key:
                status = get_chord_diatonic_status(name, key)
                if status == 'diatonic':
                    score += 0.12  # ダイアトニックコードへの加点
                elif status == 'secondary':
                    score += 0.05  # セカンダリードミナント等への微加点

            if score > best_score:
                best_score = score
                best_chord = name

        chords.append({
            "start": float(start_t),
            "end": float(end_t),
            "chord": best_chord,
            "confidence": float(best_score),
        })

    # 連続する同一コードをマージ
    merged = []
    for c in chords:
        if merged and merged[-1]["chord"] == c["chord"]:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(dict(c))

    # 低確信度のN.C.を前のコードで埋める
    for i in range(1, len(merged)):
        if merged[i]["chord"] == "N.C." and merged[i]["confidence"] < 0.3:
            merged[i]["chord"] = merged[i - 1]["chord"]

    # 再度マージ
    final = [merged[0]] if merged else []
    for c in merged[1:]:
        if final and final[-1]["chord"] == c["chord"]:
            final[-1]["end"] = c["end"]
        else:
            final.append(c)

    print(f"[chord_detector] Chroma fallback: {len(final)} chord regions")
    return final


def detect_chords(wav_path: str, beats: List[float] = None, key: str = None,
                  sr: int = 22050, hop_length: int = 512) -> List[Dict]:
    """
    コード検出のメインエントリポイント。
    BTC をプライマリ、chroma をフォールバックとして使用。

    Parameters
    ----------
    wav_path : str
        WAVファイルのパス
    beats : list[float], optional
        ビート位置(秒)。chroma fallback 時に使用。
    key : str, optional
        キー情報。ダイアトニック判定用。

    Returns
    -------
    list[dict]
        各要素: {"start": float, "end": float, "chord": str, "confidence": float}
    """
    # 1. BTC を試行
    result = detect_chords_btc(wav_path)
    if result is not None:
        # BTCでも、もしキーがあれば後でrefine可能（呼び出し側 pipeline.py で一括処理）
        return result
    
    # 2. フォールバック: chroma テンプレートマッチング
    return detect_chords_chroma(wav_path, beats=beats, key=key, sr=sr, hop_length=hop_length)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chord_detector.py <wav_path>")
        sys.exit(1)

    chords = detect_chords(sys.argv[1])
    print(f"Detected {len(chords)} chord changes:")
    for c in chords[:20]:
        src = c.get('_source', 'chroma')
        print(f"  {c['start']:.1f}-{c['end']:.1f}s: {c['chord']} ({c['confidence']:.2f}) [{src}]")
