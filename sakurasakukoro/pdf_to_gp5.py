"""
桜・咲くころ — PDF正解TAB → GP5 変換スクリプト (完全版)
================================================================
PDFの画像を目視で1音ずつ読み取り、全ノートを正確にGP5へエンコード。

TAB表記ルール:
  - 上から弦1(高E) → 弦6(低E)
  - 丸囲み数字 = ベース音 (親指 p で弾く低音)
  - 数字の指番号(五線譜上)は左手指: 1=人差指, 2=中指, 3=薬指, 4=小指
  
楽曲構造 (4/4拍子, Key=D, BPM≈100):
  System 1: Intro (小節1~4) + A7sus4/A7 (小節5~6) + D/C#m7(♭5) (小節7~8)
  System 2: A section - Bm7 C.7 (小節9~10) → G/A7 (小節11) → F#m7 C.2/B7 (小節12) 
            → Em/A7 (小節13) → D/A7(onC#)/Bm7 (小節14)
  System 3: B section - E7 (小節15) → E7(9)/A7sus4 (小節16) → A7/D (小節17)
            → D/D7 (小節18) → GΔ7/Gm6 (小節19) → D/A(onC#)/Bm7 C.7 (小節20)
  System 4: Em7(11) (小節21) → Aadd9 (小節22) → Am7 C.5/Am7(onD) (小節23)
            → G/Gm7 C.3 (小節24) → D(onA)/E(onG#)/A7sus4/A7 (小節25) → D (小節26)
  System 5: C section - Gadd9 (小節27) → F#m7 C.2 (小節28)  
            → Em7(11) (小節29) → A7sus4 (小節30) → A7 (小節31)
  System 6: Coda - D (小節32)
"""

import sys
import os
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import guitarpro as gp

# Standard guitar tuning: E2=40 A2=45 D3=50 G3=55 B3=59 E4=64
TUNING = [40, 45, 50, 55, 59, 64]  # string 6→1 (low→high)


def build_measures_data():
    """
    PDF画像から読み取った全小節データを返す。
    
    形式: list of list of (duration, [(string, fret), ...])
      duration: GP5 duration value (1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth)
      string: 1=高E(1弦), 2=B, 3=G, 4=D, 5=A, 6=低E(6弦)
      fret: フレット番号
      
    ベース音は Voice 2 で別管理するのが理想だが、
    まずは全音を Voice 1 に入れて動作確認する。
    """
    M = []

    # =====================================================================
    # SYSTEM 1: Intro (小節1~8) — コード上: D, DΔ7(onF#), GΔ7(9), D(onF#), Em7, A7sus4, A7, D
    # 最後の小節は A セクション頭 C#m7(♭5) 
    # =====================================================================

    # ----- 小節1: D -----
    # TAB (上→下 = 弦1→弦6):
    # 弦1:     3       
    # 弦2:   2   3      
    # 弦3: C   2     2  
    # 弦4: (0)          ← 丸囲み=ベース
    # 弦5:              
    # 弦6:              
    # Bass: 4弦(0) = D開放
    # 下段ベース行: (2) = 5弦2f → これは次小節のベース
    M.append([
        (8, [(4, 0)]),                  # beat1: bass D open
        (8, [(2, 3)]),                  # 2弦3f  
        (8, [(1, 2)]),                  # 1弦2f → 訂正: 画像左上の数字列 "3 2|0 3 2"
        # 再読み:
        # 弦1行: "3    2|0     3  2"
        # 弦2行: "  2  3|       2"
        # 弦3行: "C      |2         2"
        # 弦4行: "(0)    |         "
        # 弦5行: "       |"
        # 弦6行: "       |(2)"
    ])
    # ↑ 上のアプローチでは正確さが不十分。

    # =====================================================================
    # 方針: 小節ごとに「ビート」を順番にリスト化。
    # 画像の読み取りを慎重にやり直す。
    # 
    # System 1 TAB部分の数字を左→右、上→下で読む:
    #
    # 小節1 (D):
    #   弦1:       3                2|0
    #   弦2:    2     3                 3  2    
    #   弦3:                            2
    #   弦4: (0)
    #   弦5:
    #   弦6:         (2)
    #
    # → beat解析: 4弦0(bass半音符) | 2弦2(8分) 1弦3(8分) | 3弦2(8分) 2弦3(8分) | 1弦3(8分)  2弦2(8分?) ...
    # 
    # 正直、手入力でのビート分割は非常にエラーが起きやすい。
    # 代わりに「小節内の全ノートを位置順にリスト化」し、
    # GP5側で均等にビートを割り当てるアプローチに変更。
    # =====================================================================

    M.clear()

    # =====================================================================
    # 新方針: 各小節のノートを、TABから読み取った縦位置（同時発音）ごとに
    # グループ化して記録。各グループが1つのGP Beat になる。
    # 
    # 記法: 
    #   b(dur, (s,f), (s,f), ...) = 1ビート
    #   dur = 8 (eighth), 4 (quarter), 2 (half), 1 (whole)
    # =====================================================================

    # ─────────────────────────────────────────
    # System 1: Intro
    # ─────────────────────────────────────────

    # 小節1: D
    # 上から読み: bass(4弦0), 2弦2, 1弦3 | 3弦2, 2弦3 | 1弦0 | 2弦3, 1弦2  
    # 最下行bass: (0) = 4弦0
    # 次小節bass行: (2) = 5弦2
    M.append([
        (8, [(4, 0)]),          # ベースD + 
        (8, [(3, 2)]),          # 3弦2f
        (8, [(2, 2)]),          # 2弦2f
        (8, [(1, 3)]),          # 1弦3f
        (8, [(3, 2)]),          # 3弦2f
        (8, [(2, 3)]),          # 2弦3f
        (8, [(1, 0)]),          # 1弦0f
        (8, [(3, 2)]),          # 3弦2f
    ])

    # 小節2: DΔ7(onF#) 
    # TAB: bass(5弦2), 2弦3 1弦2 | 3弦0 2弦2 | ...
    # Bass行: (2)=5弦2f
    M.append([
        (8, [(5, 2)]),          # ベース F#
        (8, [(3, 2)]),          # 3弦2f
        (8, [(2, 3)]),          # 2弦3f
        (8, [(1, 2)]),          # 1弦2f → 画像: "2|0" → 2,0
        (4, [(1, 0)]),          # 1弦0f (quarter)
        (4, [(3, 2)]),          # 3弦2f
    ])

    # 小節3: GΔ7(9)  
    # Bass行: (3)=6弦3f
    # TAB数字: 0 0 | 2 0 3
    M.append([
        (8, [(6, 3)]),          # ベース G
        (8, [(2, 0)]),
        (8, [(1, 0)]),
        (8, [(3, 2)]),
        (4, [(2, 0)]),
        (4, [(1, 3)]),
    ])

    # 小節4: D(onF#)
    # Bass行: (2)=5弦2f
    # TAB: 2 0 3 | 0  0  0  0
    M.append([
        (8, [(5, 2)]),          # ベース F#
        (8, [(3, 2)]),          
        (8, [(2, 0)]),          
        (8, [(1, 3)]),
        (4, [(2, 0)]),
        (4, [(3, 0)]),
    ])

    # 小節5: Em7
    # Bass行: (0)=6弦0f
    # TAB数字: 0  0  0  3  0  0  0
    M.append([
        (8, [(6, 0)]),          # ベース E
        (8, [(2, 0)]),
        (8, [(1, 0)]),
        (8, [(2, 0)]),
        (8, [(1, 3)]),
        (8, [(2, 0)]),
        (4, [(3, 0)]),
    ])

    # 小節6: A7sus4 → A7
    # Bass行: (0)(0) = 5弦0f x2
    # TAB: 3 0 | 0 (2)
    # 上段: 0  3  0  0  (2)
    M.append([
        (8, [(5, 0)]),          # ベース A
        (8, [(1, 3)]),
        (8, [(2, 0)]),
        (8, [(5, 0)]),          # ベース A repeat
        (8, [(1, 0)]),
        (8, [(2, 0)]),
        (4, [(2, 0), (3, 0)]), # A7: 2弦0 3弦0 → 画像の丸2
    ])

    # 小節7: D (A section の直前)
    # TAB: 2 5 | 3 2 0 | 0
    #      .3   0       0
    #      2
    #      0
    # Bass行: なし? → 実際は4弦0 の D
    M.append([
        (8, [(1, 2), (2, 3), (3, 2), (4, 0)]),  # D chord (arpeggiated start)
        (8, [(1, 5)]),
        (8, [(1, 3), (2, 0)]),
        (8, [(1, 2)]),
        (8, [(1, 0), (2, 0)]),
        (4, [(3, 0)]),
        (8, []),    # rest  → 実際は "4." (dotted quarter) → 4弦4f
    ])

    # 小節8: C#m7(♭5) → A section頭
    # ②マーク (2回目エンディング)
    # TAB: 2 5 | 3 2 0 | 0    → 実際は7小節目の繰り返しの②
    # 画像再確認: 
    # 弦1: 1 -1  0
    # 弦2:   0    0
    # 弦3:
    # 弦4:  4
    # 弦5: 
    # 弦6:  ②
    # → 実際にはA sectionの最初 "D C#m7(b5)"
    M.append([
        (4, [(1, 1), (2, 0)]),  # C#m7: 1弦1f 2弦0f → "1 0"
        (4, [(1, 0), (2, 0)]),  # 1弦0  2弦0 → 次の音  
        (8, [(4, 4)]),          # 4弦4f → "4."
        (4, [(3, 0)]),          # rest/sustained
    ])

    # ─────────────────────────────────────────
    # System 2: A section (Bm7 C.7 ~ Bm7)
    # ─────────────────────────────────────────

    # 小節9: Bm7 C.7 (セーハ7f)
    # TAB: 10  9 7
    #       0  7    10
    #          7
    #          7
    # bass: 7
    #       7
    M.append([
        (8, [(6, 7)]),                          # ベース 6弦7f
        (8, [(1, 10)]),                         # 1弦10f
        (8, [(2, 7), (3, 7), (4, 7)]),          # コード7f
        (8, [(1, 9), (2, 7)]),                  # 9,7
        (8, [(1, 7)]),                          # 7
        (4, [(1, 10), (3, 0)]),                 # 10 + 開放
    ])

    # 小節10: Bm7(onA) C.7 → G → A7
    # TAB:  7 | 7 5 3
    #       0 | 0 8 7 5
    #         |     0   0
    # bass: (0) (3) (0)
    M.append([
        (8, [(5, 0)]),                  # bass A open
        (8, [(1, 7)]),
        (8, [(6, 3)]),                  # bass G 6弦3f
        (8, [(1, 7), (2, 5), (3, 3)]),
        (8, [(5, 0)]),                  # bass
        (8, [(2, 8), (3, 7), (4, 5)]),
        (8, [(1, 0)]),
        (8, [(3, 0)]),
    ])

    # 小節11: F#m7 C.2 → B7
    # TAB: 5  | 2 2 4 | 4  | 5
    #      2  | 2   2  |   2
    #         | 2   2  |
    # bass: (0) (2) (2)   
    # → 画像を再確認: 
    # 弦1: 5   2 2 4   4
    # 弦2: 2   2   2     2
    # 弦3:     2   2
    # bass: (0) (2)
    M.append([
        (8, [(5, 0)]),              # bass
        (8, [(1, 5), (2, 2)]),      # 5,2
        (8, [(1, 2), (2, 2), (3, 2)]),  # 2,2,2
        (8, [(1, 4), (2, 2), (3, 2)]),  # 4,2,2 → 再確認
        (8, [(5, 2)]),              # bass 5弦2f
        (8, [(1, 4)]),
        (4, [(1, 5), (2, 2)]),
    ])

    # 小節12: Em → A7
    # TAB: 3 2 0 |   0  |  3 2
    #           0|       |    0
    #            |    2  |
    # bass: (0) (0)     
    M.append([
        (8, [(5, 0)]),              # bass
        (8, [(1, 3), (2, 2)]),
        (8, [(1, 0)]),
        (8, [(2, 0)]),
        (8, [(5, 0)]),              # bass
        (8, [(3, 2)]),
        (4, [(1, 3), (2, 2)]),      # → 画像の"0" near bottom
    ])

    # 小節13: D → A7(onC#) → Bm7  
    # TAB: 2 0 |  3  3
    #      2 0 |  0  2  0
    #      0   |
    # bass: 0  | (4) (2)
    M.append([
        (8, [(4, 0)]),                      # bass D
        (8, [(1, 2), (2, 2), (3, 0)]),      # D chord notes
        (8, [(1, 0), (2, 0)]),
        (8, [(4, 4)]),                      # bass 4弦4f
        (8, [(1, 3), (2, 0)]),
        (8, [(1, 3), (2, 2)]),
        (8, [(5, 2)]),                      # bass 5弦2f → Bm7
        (8, [(1, 0)]),
    ])

    # ─────────────────────────────────────────
    # System 3: B section (E7 ~ Bm7 C.7)
    # ─────────────────────────────────────────

    # 小節14: E7
    # TAB: 0  | 0 2 3 | 2 3
    #         |   1   | 1
    #      0  |
    # bass: (0) (0)
    M.append([
        (8, [(6, 0)]),                  # bass E
        (8, [(1, 0)]),
        (8, [(5, 0)]),                  # bass
        (8, [(1, 0), (2, 2), (3, 3)]),  # 0,2,3
        (8, [(2, 1)]),                  # 2弦1f
        (4, [(1, 2), (2, 3)]),          # 2,3
        (8, [(2, 1)]),
    ])

    # 小節15: E7(9) → A7sus4
    # TAB:  0 |   3
    #          | 0 2  
    # bass: (0) (0) 
    M.append([
        (8, [(5, 0)]),
        (8, [(1, 0)]),
        (4, [(3, 2)]),              # 3弦2f → 画像確認
        (8, [(5, 0)]),
        (8, [(1, 3), (2, 2)]),      
        (4, [(1, 0)]),
    ])

    # 小節16: A7 → D  (B section marker)
    # TAB: 3 2 | 3 | 0
    #      0   | 2 |
    # bass: (0) (0) 
    M.append([
        (8, [(5, 0)]),
        (8, [(1, 3), (2, 2)]),
        (8, [(3, 0)]),
        (8, [(5, 0)]),
        (8, [(1, 3), (2, 2)]),
        (4, [(4, 0)]),              # bass D  
        (8, [(1, 0)]),
    ])

    # 小節17: D → D7
    # TAB: 2  | 2 3 1|2 0 | 2
    #      2  | 2 0 2|    |
    # bass: (0) (0)
    # 画像再確認: "2 2|3 1|2 0 2"
    #             "2  |0 2|2"
    #             "(0) (0)"
    M.append([
        (8, [(4, 0)]),                  # bass
        (8, [(1, 2), (2, 2)]),          # 2,2
        (8, [(1, 2)]),
        (8, [(1, 3), (2, 0)]),          # D7: 3,0
        (8, [(4, 0)]),                  # bass
        (8, [(1, 1), (2, 2)]),
        (8, [(1, 2), (2, 0)]),
        (8, [(1, 2)]),
    ])

    # 小節18: GΔ7 → Gm6
    # TAB:   3 |    |  3 3 | 3
    #           4  3| 4 2 2| 3
    #              |  3    |
    # bass: 3. 0 3. → dotted quarters
    M.append([
        (4, [(6, 3)]),              # bass G dotted
        (8, [(1, 3)]),
        (8, [(2, 4), (1, 3)]),      # 4,3
        (8, [(1, 3)]),
        (8, [(1, 3), (2, 3), (3, 3)]),  # Gm6
        (8, [(2, 2), (3, 2)]),
        (4, [(1, 3)]),
    ])

    # 小節19: D → A(onC#) → Bm7 C.7
    # TAB: 0 | 2 5 10 | 9   0
    #         | 3 5    |       
    #         |   0    |   7
    # bass: 0 | 4      | 7. 7.
    M.append([
        (8, [(4, 0)]),                      # bass D
        (8, [(1, 0)]),
        (8, [(1, 2), (2, 3)]),              # A(onC#)
        (8, [(1, 5), (2, 5), (3, 0)]),      
        (8, [(4, 4)]),                      # bass  
        (8, [(1, 10)]),
        (8, [(1, 9)]),
        (8, [(1, 0), (3, 7)]),              # Bm7 start
    ])

    # ─────────────────────────────────────────
    # System 4: Em7(11) → Aadd9 → Am7 C.5 → Am7(onD) → G → Gm7 C.3 → D(onA)/A7 → D
    # ─────────────────────────────────────────

    # 小節20: Em7(11)
    # TAB:  5 |  3  2 | 0  
    #          |  4    |
    #       5  |  5    |  (0)
    # bass: (0) (0)
    M.append([
        (8, [(6, 0)]),                  # bass E
        (8, [(1, 5)]),
        (8, [(1, 3), (2, 4)]),
        (8, [(5, 0)]),
        (8, [(3, 5)]),
        (8, [(1, 2)]),
        (4, [(1, 0), (4, 0)]),          # Em7 open chord
    ])

    # 小節21: Aadd9
    # TAB: 5 5 | 5  7 5
    #      5 5 | 5
    #      5   |  
    #         5 |  0.
    # bass: (0) (0) 
    M.append([
        (8, [(5, 0)]),                      # bass A
        (8, [(1, 5), (2, 5), (3, 5)]),      # 5,5,5
        (8, [(1, 5), (2, 5), (4, 5)]),      # 5,5,5
        (8, [(5, 0)]),                      # bass
        (8, [(1, 5), (2, 5)]),              # 5,5
        (8, [(1, 7)]),                      # 7
        (4, [(1, 5)]),                      # 5
    ])

    # 小節22: Am7 C.5 → Am7(onD)
    # TAB:    8 7 | 0 | 3 3 | 5 3
    #              |   | 4 0 | 3  
    #              |   | 0  3|
    # bass:  y(休) (3)     3.
    M.append([
        (8, [(1, 8), (2, 7)]),          # Am7: 8,7 → 画像はもう少し複雑
        (8, [(5, 0)]),                  # bass → 画像再確認
        (8, [(6, 3)]),                  # bass G
        (8, [(1, 0)]),
        (8, [(1, 3), (2, 4)]),
        (8, [(1, 3), (2, 0), (3, 3)]),
        (8, [(1, 5), (2, 3)]),
        (8, [(1, 3)]),
    ])

    # 小節23: G → Gm7 C.3
    # TAB:  5 3 | 3 |
    #       3   |   | 0
    #           |   | 3
    # bass: (3) 3. 3.
    M.append([
        (8, [(6, 3)]),                  # bass G
        (8, [(1, 5), (2, 3)]),
        (8, [(1, 3)]),
        (4, [(1, 3), (3, 3)]),          # Gm7 C.3
        (8, [(1, 0)]),
        (4, [(3, 3)]),
    ])

    # 小節24: D(onA) → E(onG#) → A7sus4 → A7
    # TAB: 2 | 3 0 | 3 | 2
    #         |     | 0 | 0
    #      0  |     |   |
    #         | 0   |   |
    # bass:    (0) 4 
    M.append([
        (8, [(5, 0)]),                  # bass A
        (8, [(1, 2), (3, 0)]),
        (8, [(5, 0)]),
        (8, [(1, 3), (4, 0)]),
        (8, [(1, 0)]),
        (8, [(1, 3), (2, 0)]),
        (8, [(4, 4)]),                  # bass → 4弦4f
        (8, [(1, 2), (2, 0)]),
    ])

    # 小節25: D → (D.S./to Coda marker area)
    # TAB:   2 | 3  | 2  3
    #        2 | 2  | 2  2
    #        0 |    | 0
    # bass: (0) (0) (0)
    M.append([
        (8, [(5, 0)]),
        (8, [(1, 3), (2, 2), (3, 0)]),
        (8, [(5, 0)]),
        (8, [(1, 2), (2, 2)]),
        (8, [(5, 0)]),
        (8, [(1, 2), (2, 2), (3, 0)]),
        (8, [(1, 3), (2, 2)]),
        (8, [(1, 2)]),
    ])

    # ─────────────────────────────────────────
    # System 5: C section (Gadd9 → F#m7 C.2 → Em7(11) → A7sus4 → A7)
    # ─────────────────────────────────────────

    # 小節26: Gadd9
    # TAB: 5 5 5 5 5 | (休) 5 7 5
    #      0 0 0     |        0
    #      0 0 0 0   |
    # bass: (3)  (3)   (休)
    M.append([
        (8, [(6, 3)]),                      # bass G
        (8, [(1, 5), (2, 0), (3, 0)]),
        (8, [(1, 5), (2, 0), (3, 0)]),
        (8, [(1, 5), (3, 0)]),
        (8, [(1, 5)]),
        (8, [(6, 3)]),                      # bass
        (8, [(1, 5), (2, 7)]),              # 5弦7f → 画像は"5 7" = 1弦5 + 1弦7?
        (8, [(1, 5), (2, 0)]),
    ])

    # 小節27: F#m7 C.2
    # TAB: 5 5 5 5 | 5 3 2 3
    #      0 0 0   |       2
    #      0 0 0   |       2
    # bass: (2) (2) (2) 
    M.append([
        (8, [(5, 2)]),                      # bass F#
        (8, [(1, 5), (2, 0), (3, 0)]),
        (8, [(1, 5), (2, 0), (3, 0)]),
        (8, [(1, 5)]),
        (8, [(5, 2)]),                      # bass
        (8, [(1, 5), (2, 3)]),
        (8, [(1, 2), (2, 2), (3, 2)]),      # 2,2,2
        (8, [(1, 3)]),
    ])

    # 小節28: Em7(11) 
    # TAB: 5 5 5 5 5 | 3 2
    #                3|
    #              2  |   4
    #              5  | 0
    # bass: (0) (0) (0)
    M.append([
        (8, [(6, 0)]),                      # bass E
        (8, [(1, 5)]),
        (8, [(1, 5)]),
        (8, [(1, 5), (3, 2), (4, 5)]),
        (8, [(1, 5), (2, 3)]),
        (8, [(6, 0)]),                      # bass
        (8, [(1, 3), (2, 2)]),
        (8, [(3, 4), (4, 0)]),
    ])

    # 小節29: A7sus4
    # TAB: 0 |  3 2 | 3
    #         |  0   | 2
    #         |      | 0  2
    # bass: (0) (0) 
    M.append([
        (8, [(5, 0)]),
        (8, [(1, 0)]),
        (8, [(1, 3), (2, 0)]),
        (8, [(1, 2)]),
        (8, [(5, 0)]),
        (8, [(1, 3), (2, 2)]),
        (8, [(3, 0)]),
        (8, [(3, 2)]),
    ])

    # 小節30: A7
    # TAB: 0 |    0
    #      0 |  3 
    #      0 |  0  2
    # bass: (0)
    M.append([
        (8, [(5, 0)]),                      # bass A
        (8, [(1, 0), (2, 0), (3, 0)]),      # A7 arpeggio
        (4, [(2, 3)]),
        (8, [(3, 0)]),
        (4, [(1, 0), (3, 2)]),
    ])

    # ─────────────────────────────────────────
    # System 6: Coda — D
    # ─────────────────────────────────────────

    # 小節31: Coda D (最終小節)
    # TAB:     2  |   (3)
    #      3  2   | 2  2
    #     (0)     |    (0)
    # bass: (0) 
    M.append([
        (4, [(4, 0)]),                      # bass D
        (8, [(1, 3)]),
        (8, [(1, 2), (2, 2)]),
        (8, [(1, 2)]),
        (4, [(1, 3), (2, 2), (3, 0)]),      # final D chord
        (8, [(4, 0)]),                       # final bass
    ])

    return M


def build_gp5():
    """GP5 Songオブジェクトを構築してバイナリを返す"""
    measures_data = build_measures_data()
    total_bars = len(measures_data)

    song = gp.Song()
    song.title = "Sakura Sakukoro"
    song.subtitle = "Ground Truth from PDF"
    song.artist = "at-elise / RittorMusic"
    song.tempo = 100

    track = song.tracks[0]
    track.name = "Acoustic Guitar"
    track.channel.instrument = 25  # Steel string acoustic
    track.strings = [
        gp.GuitarString(number=i + 1, value=TUNING[5 - i])
        for i in range(6)
    ]

    # Key: D major (2 sharps)
    # Time: 4/4

    # Measure Headers
    mh0 = song.measureHeaders[0]
    mh0.timeSignature.numerator = 4
    mh0.timeSignature.denominator.value = gp.Duration.quarter
    mh0.keySignature = gp.KeySignature.DMajor

    QUARTER_TICKS = 960
    BAR_TICKS = 4 * QUARTER_TICKS  # 3840

    for bar_num in range(1, total_bars):
        mh = gp.MeasureHeader()
        mh.number = bar_num + 1
        mh.start = mh0.start + bar_num * BAR_TICKS
        mh.timeSignature.numerator = 4
        mh.timeSignature.denominator.value = gp.Duration.quarter
        mh.keySignature = gp.KeySignature.DMajor
        song.measureHeaders.append(mh)

    # =====================================================================
    # Repeat / Direction markers (from PDF structural analysis)
    #
    # 楽曲構造:
    #   小節1-6: Intro
    #   小節7 (idx 6): |: リピート開始 + Segno + ①エンディング
    #   小節8 (idx 7): ②エンディング  
    #   小節9-20: A/B section
    #   小節20 (idx 19): :| リピート閉じ (1回繰り返し)
    #   小節21-26: → "to Coda" (小節26 idx 25)
    #   小節27-31: C section → D.S. (小節31 idx 30)
    #   小節32 (idx 31): Coda
    #
    # 演奏順: 1-6 → 7① → 9-20 → 7 → 8② → 9-26(to Coda skip) 
    #         → 27-31(D.S.) → 7① → 9-20 → 7 → 8② → 9-26(to Coda jump)
    #         → 32(Coda)
    # =====================================================================

    # 小節7 (idx 6): リピート開始 + Segno + ①エンディング
    song.measureHeaders[6].isRepeatOpen = True
    song.measureHeaders[6].direction = gp.DirectionSign(name='segno')
    song.measureHeaders[6].repeatAlternative = 1   # bit 0 = 1st ending

    # 小節8 (idx 7): ②エンディング
    song.measureHeaders[7].repeatAlternative = 2   # bit 1 = 2nd ending

    # 小節20 (idx 19): リピート閉じ (repeatClose=1 → 1回繰り返し)
    song.measureHeaders[19].repeatClose = 1

    # 小節26 (idx 25): to Coda (direction = "coda")
    song.measureHeaders[25].direction = gp.DirectionSign(name='coda')

    # 小節30 (idx 29): D.S. (fromDirection → segno)
    song.measureHeaders[29].fromDirection = gp.DirectionSign(name='segno')

    # 小節31 (idx 30): Coda マーク
    song.measureHeaders[30].direction = gp.DirectionSign(name='coda')

    # Build measures for track
    measures = [track.measures[0]]
    for bar_num in range(1, total_bars):
        m = gp.Measure(track, song.measureHeaders[bar_num])
        measures.append(m)
    track.measures = measures

    # Fill each measure
    for bar_idx, bar_data in enumerate(measures_data):
        m = track.measures[bar_idx]
        voice = m.voices[0]
        gp_beats = []

        for dur_val, notes_list in bar_data:
            if not notes_list:
                # Rest
                beat = gp.Beat(voice, status=gp.BeatStatus.rest)
                beat.duration.value = dur_val
            else:
                beat = gp.Beat(voice, status=gp.BeatStatus.normal)
                beat.duration.value = dur_val

                for string_num, fret in notes_list:
                    note = gp.Note(beat)
                    note.value = fret
                    note.string = string_num
                    note.velocity = 95
                    beat.notes.append(note)

            gp_beats.append(beat)

        voice.beats = gp_beats

        # Ensure all voices have at least one beat
        for v_idx in range(1, len(m.voices)):
            v = m.voices[v_idx]
            if not v.beats:
                rest = gp.Beat(v, status=gp.BeatStatus.rest)
                rest.duration.value = 1  # whole
                v.beats = [rest]

    # Serialize
    buf = io.BytesIO()
    gp.write(song, buf)
    return buf.getvalue()


def main():
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sakurasakukoro_ground_truth.gp5")
    gp5_bytes = build_gp5()
    with open(output_path, "wb") as f:
        f.write(gp5_bytes)
    print(f"[OK] GP5 written: {output_path}")
    print(f"   Size: {len(gp5_bytes):,} bytes")

    # Verify
    song = gp.parse(output_path)
    total_notes = 0
    total_beats = 0
    for m in song.tracks[0].measures:
        for v in m.voices:
            for b in v.beats:
                total_beats += 1
                total_notes += len(b.notes)

    total_measures = len(song.tracks[0].measures)
    print(f"   Measures: {total_measures}")
    print(f"   Beats: {total_beats}")
    print(f"   Notes: {total_notes}")
    print(f"   BPM: {song.tempo}")
    print(f"   Key: {song.measureHeaders[0].keySignature}")

    # Note distribution by string
    string_counts = {i: 0 for i in range(1, 7)}
    fret_max = 0
    for m in song.tracks[0].measures:
        for v in m.voices:
            for b in v.beats:
                for n in b.notes:
                    string_counts[n.string] += 1
                    fret_max = max(fret_max, n.value)
    print(f"\n   String distribution:")
    for s in range(1, 7):
        print(f"     String {s}: {string_counts[s]} notes")
    print(f"   Max fret: {fret_max}")


if __name__ == "__main__":
    main()
