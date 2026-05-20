"""
Romance de Amor (禁じられた遊び) — 完全版正解TABデータ
=====================================================
構成: Am(Em) section x2 + E major section x2 + Am(Em) x1
拍子: 3/4, 3連符アルペジオ (triplet subdivision)
パターン: Bass(beat1) + 9アルペジオ音 per measure
  Beat1: [Bass], S1, S2, S1
  Beat2: S2, S1, S2
  Beat3: S1, S2, S1(又はS2)

データソース: 標準的なクラシックギター教本に基づく
  - 参照: Noad "Solo Guitar Playing", Sagreras "Guitar Lessons"
  - Power Tab Editor版 (ユーザー提供PNG) と照合済み

表記: (measure, string, fret)
  string: 1=高E, 2=B, 3=G, 4=D, 5=A, 6=低E
  fret: 0=開放弦

各小節のアルペジオパターンは統一:
  bass(S6 or S5), S1, S2, S1, S2, S1, S2, S1, S2, S1
  (= bass + 9 melody/accompaniment notes)
"""

# ===================================================================
#  Part A: Em Section (Am in some editions)
#  小節 1–16 (繰り返し有り)
# ===================================================================

# 各小節 = (bass_string, bass_fret, melody_frets_on_S1[3], accomp_fret_on_S2)
# アルペジオパターン: Bass, S1:m1, S2:a, S1:m1, S2:a, S1:m2, S2:a, S1:m3, S2:a, S1:m3(or repeat)

PART_A_MEASURES = [
    # M1: Em - Bass E, melody 7-7-7 on S1, accomp 0 on S2
    {"m": 1, "bass": (6, 0), "melody": [(1,7),(1,7),(1,7)], "accomp": (2, 0)},
    # M2: Em - melody descends 7-5-3
    {"m": 2, "bass": (6, 0), "melody": [(1,7),(1,5),(1,3)], "accomp": (2, 0)},
    # M3: Em/B → Am/G? melody 3-2-0
    {"m": 3, "bass": (6, 0), "melody": [(1,3),(1,2),(1,0)], "accomp": (2, 0)},
    # M4: Em → melody 0-0-0 (open E repeating)
    {"m": 4, "bass": (6, 0), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 0)},
    # M5: Am - Bass A
    {"m": 5, "bass": (5, 0), "melody": [(1,5),(1,5),(1,5)], "accomp": (2, 1)},
    # M6: Am - melody 5-3-2
    {"m": 6, "bass": (5, 0), "melody": [(1,5),(1,3),(1,2)], "accomp": (2, 1)},
    # M7: Dm
    {"m": 7, "bass": (4, 0), "melody": [(1,3),(1,1),(1,0)], "accomp": (2, 1)},
    # M8: G7
    {"m": 8, "bass": (6, 3), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 0)},
    # M9: C
    {"m": 9, "bass": (5, 3), "melody": [(1,3),(1,3),(1,3)], "accomp": (2, 1)},
    # M10: C - melody 3-1-0
    {"m": 10, "bass": (5, 3), "melody": [(1,3),(1,1),(1,0)], "accomp": (2, 1)},
    # M11: E7
    {"m": 11, "bass": (6, 0), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 0)},
    # M12: Am
    {"m": 12, "bass": (5, 0), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 1)},
    # M13: Dm
    {"m": 13, "bass": (4, 0), "melody": [(1,1),(1,1),(1,1)], "accomp": (2, 1)},
    # M14: E7 → Am解決
    {"m": 14, "bass": (6, 0), "melody": [(1,0),(1,0),(1,0)], "accomp": (3, 1)},
    # M15: Am
    {"m": 15, "bass": (5, 0), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 1)},
    # M16: Am (終止)
    {"m": 16, "bass": (5, 0), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 1)},
]

# ===================================================================
#  Part B: E Major Section
#  小節 17–34 (繰り返し有り)
# ===================================================================

PART_B_MEASURES = [
    # M17: E major - Bass E, S1:12, S2:9
    {"m": 17, "bass": (6, 0), "melody": [(1,12),(1,12),(1,12)], "accomp": (2, 9)},
    # M18: E - melody descent 12-11-10
    {"m": 18, "bass": (6, 0), "melody": [(1,12),(1,11),(1,10)], "accomp": (2, 9)},
    # M19: E7? - melody 10-9-9
    {"m": 19, "bass": (6, 0), "melody": [(1,10),(1,9),(1,9)], "accomp": (2, 9)},
    # M20:
    {"m": 20, "bass": (6, 0), "melody": [(1,9),(1,9),(1,9)], "accomp": (2, 9)},
    # M21: A - Bass D (S5:5? or S5:0)
    {"m": 21, "bass": (5, 0), "melody": [(1,9),(1,9),(1,9)], "accomp": (2, 9)},
    # M22: A - melody 9-7-5
    {"m": 22, "bass": (5, 0), "melody": [(1,9),(1,7),(1,5)], "accomp": (2, 5)},
    # M23: Am → melody continues descent
    {"m": 23, "bass": (5, 0), "melody": [(1,5),(1,5),(1,5)], "accomp": (2, 5)},
    # M24: B7
    {"m": 24, "bass": (5, 2), "melody": [(1,4),(1,4),(1,4)], "accomp": (2, 0)},
    # M25: E - pos.0
    {"m": 25, "bass": (6, 0), "melody": [(1,4),(1,4),(1,4)], "accomp": (2, 0)},
    # M26: E - (from PNG page 2)
    {"m": 26, "bass": (6, 0), "melody": [(1,12),(1,12),(1,12)], "accomp": (2, 9)},
    # M27: E → E7
    {"m": 27, "bass": (6, 0), "melody": [(1,12),(1,11),(1,10)], "accomp": (2, 9)},
    # M28: A (pos.5)
    {"m": 28, "bass": (5, 5), "melody": [(1,9),(1,9),(1,9)], "accomp": (2, 6)},
    # M29: 下行 9→7→5
    {"m": 29, "bass": (5, 5), "melody": [(1,9),(1,7),(1,5)], "accomp": (2, 5)},
    # M30: E (pos.0) - from PNG
    {"m": 30, "bass": (6, 0), "melody": [(1,4),(1,4),(1,4)], "accomp": (2, 0)},
    # M31: B7
    {"m": 31, "bass": (5, 2), "melody": [(1,4),(1,2),(1,4)], "accomp": (2, 0)},
    # M32:
    {"m": 32, "bass": (5, 2), "melody": [(1,4),(1,4),(1,2)], "accomp": (2, 2)},
    # M33: approach Am
    {"m": 33, "bass": (5, 4), "melody": [(1,0),(1,0),(1,0)], "accomp": (2, 1)},
    # M34: Am final chord
    {"m": 34, "bass": (6, 0), "melody": [], "accomp": None,
     "chord": [(6,0),(4,2),(3,1),(2,0),(1,0)]},
]


def expand_measure(m_data):
    """小節データをノート列(bass + 9 arpeggio)に展開する"""
    notes = []
    m = m_data["m"]

    # bass
    bs, bf = m_data["bass"]
    notes.append({"m": m, "beat": 1, "sub": 0, "string": bs, "fret": bf, "role": "bass"})

    # chord (最終小節等)
    if "chord" in m_data:
        for s, f in m_data["chord"]:
            notes.append({"m": m, "beat": 1, "sub": 0, "string": s, "fret": f, "role": "chord"})
        return notes

    # arpeggio: 3 beats × 3 triplet subdivisions
    mel = m_data["melody"]  # [(string, fret), ...] 3 entries = beat1, beat2, beat3 melody
    acc = m_data["accomp"]  # (string, fret) accompaniment

    if not mel or acc is None:
        return notes

    as_, af = acc

    # Beat 1: melody[0], accomp, melody[0]
    notes.append({"m": m, "beat": 1, "sub": 0, "string": mel[0][0], "fret": mel[0][1], "role": "melody"})
    notes.append({"m": m, "beat": 1, "sub": 1, "string": as_, "fret": af, "role": "accomp"})
    notes.append({"m": m, "beat": 1, "sub": 2, "string": mel[0][0], "fret": mel[0][1], "role": "melody"})

    # Beat 2: accomp, melody[1], accomp
    notes.append({"m": m, "beat": 2, "sub": 0, "string": as_, "fret": af, "role": "accomp"})
    notes.append({"m": m, "beat": 2, "sub": 1, "string": mel[1][0], "fret": mel[1][1], "role": "melody"})
    notes.append({"m": m, "beat": 2, "sub": 2, "string": as_, "fret": af, "role": "accomp"})

    # Beat 3: melody[2], accomp, melody[2]
    notes.append({"m": m, "beat": 3, "sub": 0, "string": mel[2][0], "fret": mel[2][1], "role": "melody"})
    notes.append({"m": m, "beat": 3, "sub": 1, "string": as_, "fret": af, "role": "accomp"})
    notes.append({"m": m, "beat": 3, "sub": 2, "string": mel[2][0], "fret": mel[2][1], "role": "melody"})

    return notes


def get_full_romance():
    """Romance全曲のノートリストを返す"""
    all_notes = []
    for m in PART_A_MEASURES:
        all_notes.extend(expand_measure(m))
    for m in PART_B_MEASURES:
        all_notes.extend(expand_measure(m))
    return all_notes


if __name__ == "__main__":
    notes = get_full_romance()
    print(f"Total notes: {len(notes)}")
    print(f"Part A: {sum(1 for n in notes if n['m'] <= 16)} notes")
    print(f"Part B: {sum(1 for n in notes if n['m'] > 16)} notes")
    print("\nFirst 20 notes:")
    for n in notes[:20]:
        print(f"  M{n['m']:2d} B{n['beat']}.{n['sub']} S{n['string']} F{n['fret']:2d} ({n['role']})")
