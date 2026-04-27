"""
pdf_renderer.py — MusicXMLからTAB譜PDFを生成
================================================
reportlab を使用して、MusicXML (tab.musicxml) を解析し、
ギターTAB譜のPDFを直接生成する。
AlphaTabやブラウザのレンダリングに一切依存しない。
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# フォント登録（日本語対応）
_FONT_REGISTERED = False
def _ensure_fonts():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return

    # 方法1: reportlab 内蔵 CID フォント（最も確実）
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
        _FONT_REGISTERED = True
        return
    except Exception:
        pass

    # 方法2: OS のフォントファイル
    candidates = [
        ("C:/Windows/Fonts/meiryo.ttc", "Meiryo", 0),
        ("C:/Windows/Fonts/msgothic.ttc", "MSGothic", 0),
        ("C:/Windows/Fonts/arial.ttf", "Arial", None),
    ]
    for path, name, subfontIndex in candidates:
        if os.path.exists(path):
            try:
                if subfontIndex is not None:
                    pdfmetrics.registerFont(TTFont(name, path, subfontIndex=subfontIndex))
                else:
                    pdfmetrics.registerFont(TTFont(name, path))
            except Exception:
                pass
    _FONT_REGISTERED = True


def _get_jp_font():
    """利用可能な日本語フォント名を返す"""
    for name in ["HeiseiKakuGo-W5", "Meiryo", "MSGothic"]:
        try:
            pdfmetrics.getFont(name)
            return name
        except KeyError:
            pass
    return "Helvetica"


# TAB譜のレイアウト定数
PAGE_W, PAGE_H = A4  # 595.28 x 841.89 pt
MARGIN_LEFT = 25 * mm
MARGIN_RIGHT = 15 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOTTOM = 20 * mm
STAFF_LINE_SPACING = 8  # 弦間の間隔(pt)
STAFF_HEIGHT = STAFF_LINE_SPACING * 5  # 6弦なので5間隔
SYSTEM_SPACING = 48  # 段間の余白(pt) — リズムステム描画スペース含む
NOTES_AREA_WIDTH = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
BAR_MIN_WIDTH = 80  # 1小節の最小幅(pt)

# 弦名ラベル
STRING_LABELS = ["e", "B", "G", "D", "A", "E"]  # 1弦(上)→6弦(下)


def musicxml_to_pdf(musicxml_path: str, output_path: str, title: str = "Guitar TAB") -> str:
    """
    MusicXMLファイルを読み込み、TAB譜PDFを生成する。
    
    Parameters
    ----------
    musicxml_path : str
        入力のMusicXMLファイルパス
    output_path : str
        出力PDFファイルパス
    title : str
        曲のタイトル
    
    Returns
    -------
    str
        生成されたPDFファイルパス
    """
    _ensure_fonts()
    jp_font = _get_jp_font()
    
    # MusicXMLをパース
    tree = ET.parse(musicxml_path)
    root = tree.getroot()
    
    # タイトル取得
    work_title = root.find(".//work-title")
    if work_title is not None and work_title.text:
        title = work_title.text
    
    # BPM取得
    bpm_el = root.find(".//sound[@tempo]")
    bpm = int(bpm_el.get("tempo")) if bpm_el is not None else 120
    
    # 拍子取得
    beats_el = root.find(".//time/beats")
    beat_type_el = root.find(".//time/beat-type")
    time_sig = "4/4"
    if beats_el is not None and beat_type_el is not None:
        time_sig = f"{beats_el.text}/{beat_type_el.text}"
    
    # 全小節のデータを抽出
    measures = _parse_measures(root)
    
    # PDF生成
    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle(title)
    
    # === 1ページ目ヘッダー ===
    y = PAGE_H - MARGIN_TOP
    
    # タイトル
    c.setFont(jp_font, 16)
    display_title = title
    if c.stringWidth(display_title, jp_font, 16) > PAGE_W - 80:
        c.setFont(jp_font, 12)
        if c.stringWidth(display_title, jp_font, 12) > PAGE_W - 80:
            display_title = display_title[:40] + "..."
    c.drawCentredString(PAGE_W / 2, y, display_title)
    y -= 20
    
    # メタ情報
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_LEFT, y, f"♩= {bpm}    {time_sig}")
    c.drawRightString(PAGE_W - MARGIN_RIGHT, y, "NextChord SoloTab")
    y -= 25
    
    # === 小節を段に割り当て ===
    bars_per_row = 4
    rows = []
    for i in range(0, len(measures), bars_per_row):
        rows.append(measures[i:i + bars_per_row])
    
    row_height = STAFF_HEIGHT + SYSTEM_SPACING
    
    for row_idx, row in enumerate(rows):
        # ページの残りスペースをチェック
        if y - row_height < MARGIN_BOTTOM:
            c.showPage()
            y = PAGE_H - MARGIN_TOP
            # ページ番号
            c.setFont("Helvetica", 8)
            c.drawRightString(PAGE_W - MARGIN_RIGHT, MARGIN_BOTTOM - 10,
                              f"- {c.getPageNumber()} -")
        
        # 段の描画
        _draw_system(c, MARGIN_LEFT, y, NOTES_AREA_WIDTH, row, row_idx == 0, jp_font)
        y -= row_height
    
    # 最終ページのページ番号
    c.setFont("Helvetica", 8)
    c.drawRightString(PAGE_W - MARGIN_RIGHT, MARGIN_BOTTOM - 10,
                      f"- {c.getPageNumber()} -")
    
    c.save()
    return output_path


def _parse_measures(root: ET.Element) -> list:
    """MusicXMLからすべての小節データを抽出する"""
    measures = []
    part = root.find(".//part")
    if part is None:
        return measures
    
    # divisionsを取得（MusicXMLのquarter noteあたりのdivision数）
    divisions = 12  # デフォルト
    div_el = root.find(".//attributes/divisions")
    if div_el is not None and div_el.text:
        divisions = int(div_el.text)
    
    for measure_el in part.findall("measure"):
        measure_num = measure_el.get("number", "?")
        notes = []
        chord_name = None
        
        # divisionsの更新（小節内で変わる場合）
        local_div = measure_el.find("attributes/divisions")
        if local_div is not None and local_div.text:
            divisions = int(local_div.text)
        
        # コード(Harmony)
        harmony = measure_el.find("harmony")
        if harmony is not None:
            root_step = harmony.find("root/root-step")
            root_alter = harmony.find("root/root-alter")
            kind = harmony.find("kind")
            if root_step is not None:
                chord_name = root_step.text or ""
                if root_alter is not None and root_alter.text:
                    alter_val = int(root_alter.text)
                    chord_name += "#" if alter_val > 0 else "b"
                if kind is not None and kind.text:
                    kind_map = {
                        "minor": "m", "dominant": "7", "minor-seventh": "m7",
                        "major": "", "major-seventh": "M7", "diminished": "dim",
                        "augmented": "aug", "suspended-fourth": "sus4",
                        "suspended-second": "sus2",
                    }
                    chord_name += kind_map.get(kind.text, "")
        
        # ノート
        for note_el in measure_el.findall("note"):
            rest = note_el.find("rest")
            if rest is not None:
                continue
            
            technical = note_el.find(".//technical")
            if technical is None:
                continue
            
            string_el = technical.find("string")
            fret_el = technical.find("fret")
            if string_el is None or fret_el is None:
                continue
            
            string_num = int(string_el.text)  # 1=1弦(高E), 6=6弦(低E)
            fret_num = int(fret_el.text)
            
            is_chord = note_el.find("chord") is not None
            
            # duration情報
            dur_el = note_el.find("duration")
            dur_divs = int(dur_el.text) if dur_el is not None and dur_el.text else divisions
            type_el = note_el.find("type")
            note_type = type_el.text if type_el is not None and type_el.text else "quarter"
            
            # テクニック情報
            technique = "normal"
            harmonic = technical.find("harmonic")
            if harmonic is not None:
                technique = "harmonic"
            elif note_el.find(".//ghost") is not None:
                technique = "ghost"
            
            notes.append({
                "string": string_num,
                "fret": fret_num,
                "chord": is_chord,
                "technique": technique,
                "duration": dur_divs,
                "type": note_type,
                "divisions": divisions,
            })
        
        measures.append({
            "number": measure_num,
            "notes": notes,
            "chord_name": chord_name,
        })
    
    return measures


def _draw_system(c: canvas.Canvas, x: float, y: float, width: float,
                 measures: list, is_first: bool, jp_font: str):
    """1段(=複数小節)のTAB譜を描画"""
    
    # 弦名ラベル用スペース
    label_width = 15
    staff_x = x + label_width
    staff_width = width - label_width
    
    # 6本の弦線を描画
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    for i in range(6):
        line_y = y - i * STAFF_LINE_SPACING
        c.line(staff_x, line_y, staff_x + staff_width, line_y)
    
    # 弦名ラベル
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    for i, label in enumerate(STRING_LABELS):
        c.drawRightString(staff_x - 3, y - i * STAFF_LINE_SPACING - 2.5, label)
    c.setFillColorRGB(0, 0, 0)
    
    # 小節線と小節内ノートの描画
    num_bars = len(measures)
    if num_bars == 0:
        return
    
    bar_width = staff_width / num_bars
    
    for bar_idx, measure in enumerate(measures):
        bar_x = staff_x + bar_idx * bar_width
        
        # 小節線（左端）
        c.setLineWidth(0.8)
        c.line(bar_x, y, bar_x, y - STAFF_HEIGHT)
        
        # コード名
        if measure.get("chord_name"):
            c.setFont(jp_font, 9)
            c.setFillColorRGB(0.1, 0.1, 0.6)
            c.drawString(bar_x + 3, y + 12, measure["chord_name"])
            c.setFillColorRGB(0, 0, 0)
        
        # 小節番号（最初の段のみ or 各段の最初の小節）
        if bar_idx == 0:
            c.setFont("Helvetica", 6)
            c.setFillColorRGB(0.6, 0, 0)
            c.drawString(bar_x + 2, y + 5, str(measure["number"]))
            c.setFillColorRGB(0, 0, 0)
        
        # ノートの描画
        _draw_notes_in_bar(c, bar_x, y, bar_width, measure["notes"])
    
    # 最終小節線（右端）
    c.setLineWidth(0.8)
    end_x = staff_x + staff_width
    c.line(end_x, y, end_x, y - STAFF_HEIGHT)


def _draw_notes_in_bar(c: canvas.Canvas, bar_x: float, y: float,
                       bar_width: float, notes: list):
    """1小節内のノートを描画（フレット番号 + リズムステム）
    
    音楽理論に基づく配置ルール:
    - ノートのX座標は duration（拍内の位置）に比例させる
    - リズムステムは拍（beat）単位でグループ化して連桁（beam）
    - 全音符=ステムなし, 2分=ステム+白丸, 4分=ステムのみ
    - 8分=1本ビーム, 16分=2本ビーム
    """
    if not notes:
        return
    
    # ノートをグループ分け（chord=Trueは前のノートと同時発音）
    groups = []
    current_group = []
    for note in notes:
        if note["chord"] and current_group:
            current_group.append(note)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [note]
    if current_group:
        groups.append(current_group)
    
    # --- Duration比例配置 ---
    # 各グループのtick位置を計算し、小節幅に比例配置する
    divisions = notes[0].get("divisions", 12) if notes else 12
    beats_per_bar = 4  # TODO: 拍子情報をmeasureから渡す
    bar_total_ticks = divisions * beats_per_bar
    
    # 各グループの開始tick位置を累積計算
    tick_positions = []
    current_tick = 0
    for group in groups:
        tick_positions.append(current_tick)
        dur = group[0].get("duration", divisions)
        current_tick += dur
    
    padding = 8  # 小節端からの余白
    usable_width = bar_width - padding * 2
    
    # リズムステムの基準位置（タブ譜6弦線の4pt下）
    stem_base_y = y - STAFF_HEIGHT - 4
    group_render_data = []  # 描画用データ
    
    for g_idx, group in enumerate(groups):
        # Duration比例のX座標
        tick = tick_positions[g_idx]
        ratio = tick / max(bar_total_ticks, 1)
        note_x = bar_x + padding + ratio * usable_width
        
        # グループのnote typeを取得（先頭ノートから）
        note_type = group[0].get("type", "quarter")
        group_dur = group[0].get("duration", divisions)
        
        # --- フレット番号の描画 ---
        for note in group:
            string_idx = note["string"] - 1  # 0-indexed (0=1弦=上)
            note_y = y - string_idx * STAFF_LINE_SPACING
            fret_text = str(note["fret"])
            
            # フレット番号の背景（白で弦線を消す）
            text_width = max(7, len(fret_text) * 5.5)
            c.setFillColorRGB(1, 1, 1)
            c.rect(note_x - text_width / 2, note_y - 4.5, text_width, 9, fill=1, stroke=0)
            
            # フレット番号
            c.setFont("Helvetica-Bold", 8)
            if note.get("technique") == "harmonic":
                c.setFillColorRGB(0, 0.5, 0)
            elif note.get("technique") == "ghost":
                c.setFillColorRGB(0.5, 0.5, 0.5)
            else:
                c.setFillColorRGB(0, 0, 0)
            c.drawCentredString(note_x, note_y - 3, fret_text)
            c.setFillColorRGB(0, 0, 0)
        
        # グループ描画データを保存
        group_render_data.append({
            "x": note_x,
            "type": note_type,
            "duration": group_dur,
            "tick": tick,
        })
    
    # --- リズムステム＆ビームの描画 ---
    _draw_rhythm_notation(c, group_render_data, stem_base_y, divisions)


def _draw_rhythm_notation(c: canvas.Canvas, groups: list, base_y: float, divisions: int):
    """
    TAB譜のリズム表記を音楽理論に基づいて描画する。
    
    音楽理論ルール:
    1. 全音符(whole): ステムなし
    2. 2分音符(half): ステムのみ（下向き縦線）
    3. 4分音符(quarter): ステムのみ（下向き縦線）
    4. 8分音符(eighth): ステム + 旗1本。連続する場合は1本の水平ビームで連結
    5. 16分音符(16th): ステム + 旗2本。連続する場合は2本の水平ビームで連結
    6. ビーム（連桁）は1拍（= divisions ticks）の範囲内でのみ連結する
    """
    if not groups:
        return
    
    stem_len = 12
    stem_top = base_y
    stem_bottom = base_y - stem_len
    beam_thickness = 2.0
    beam_spacing = 3.5  # 1本目と2本目のビーム間隔
    
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    
    # --- Step 1: 全グループにステムを描画 ---
    for g in groups:
        x = g["x"]
        ntype = g["type"]
        c.setLineWidth(0.8)
        
        if ntype == "whole":
            # 全音符: ステムなし（何も描かない）
            pass
        elif ntype == "half":
            # 2分音符: ステムのみ
            c.line(x, stem_top, x, stem_bottom)
        else:
            # 4分・8分・16分: ステムを描画
            c.line(x, stem_top, x, stem_bottom)
    
    # --- Step 2: 8分・16分音符を拍（beat）単位でグループ化 ---
    divs = max(1, divisions)
    beam_groups = []
    current_beam = []
    
    for g in groups:
        ntype = g["type"]
        tick = g.get("tick", 0)
        beat_idx = int(tick / divs)
        
        if ntype in ("eighth", "16th"):
            # 拍が変わったら新グループ
            if current_beam and current_beam[-1]["_beat"] != beat_idx:
                beam_groups.append(current_beam)
                current_beam = []
            g["_beat"] = beat_idx
            current_beam.append(g)
        else:
            # 全音符・2分・4分はビームを切る
            if current_beam:
                beam_groups.append(current_beam)
                current_beam = []
    if current_beam:
        beam_groups.append(current_beam)
    
    # --- Step 3: 各ビームグループの描画 ---
    for bg in beam_groups:
        if len(bg) == 1:
            # 孤立した8分/16分 → 旗（flag）を描く
            _draw_flag(c, bg[0]["x"], stem_bottom, bg[0]["type"])
        else:
            # 複数ノート → 水平ビーム（beam）で連結
            start_x = bg[0]["x"]
            end_x = bg[-1]["x"]
            
            # 1本目のビーム（8分音符レベル）
            c.setLineWidth(beam_thickness)
            c.line(start_x, stem_bottom, end_x, stem_bottom)
            
            # 2本目のビーム（16分音符レベル）
            # 16分音符が連続する区間のみ2本目を描く
            _draw_16th_sub_beams(c, bg, stem_bottom, beam_spacing, beam_thickness)
    
    c.setLineWidth(0.5)  # リセット


def _draw_flag(c: canvas.Canvas, x: float, stem_bottom: float, note_type: str):
    """
    孤立した8分/16分音符の旗（flag）を描画する。
    旗は右向きの短い曲線（近似として斜め線）。
    """
    c.setLineWidth(1.2)
    flag_len = 6
    flag_drop = 4  # 旗の垂れ下がり
    
    # 1本目の旗
    p = c.beginPath()
    p.moveTo(x, stem_bottom)
    p.curveTo(x + flag_len * 0.3, stem_bottom - flag_drop * 0.2,
              x + flag_len * 0.7, stem_bottom + flag_drop * 0.3,
              x + flag_len, stem_bottom + flag_drop)
    c.drawPath(p, stroke=1, fill=0)
    
    if note_type == "16th":
        # 2本目の旗（3.5pt上にずらす）
        offset = 3.5
        p2 = c.beginPath()
        p2.moveTo(x, stem_bottom + offset)
        p2.curveTo(x + flag_len * 0.3, stem_bottom + offset - flag_drop * 0.2,
                   x + flag_len * 0.7, stem_bottom + offset + flag_drop * 0.3,
                   x + flag_len, stem_bottom + offset + flag_drop)
        c.drawPath(p2, stroke=1, fill=0)


def _draw_16th_sub_beams(c: canvas.Canvas, bg: list, stem_bottom: float,
                         beam_spacing: float, beam_thickness: float):
    """
    ビームグループ内で16分音符が連続する区間に2本目のサブビームを描画する。
    
    音楽理論ルール:
    - 8分音符の間に挟まれた単独16分 → 短いサブビーム（自ノート側に寄せる）
    - 連続16分 → 区間全体に2本目ビーム
    """
    c.setLineWidth(beam_thickness)
    sub_y = stem_bottom + beam_spacing
    
    i = 0
    while i < len(bg):
        if bg[i]["type"] == "16th":
            # 連続する16分音符の区間を見つける
            j = i
            while j < len(bg) and bg[j]["type"] == "16th":
                j += 1
            
            if j - i == 1:
                # 単独16分 → 短いサブビーム
                x = bg[i]["x"]
                if i > 0:
                    # 前のノートの方向に寄せる
                    prev_x = bg[i - 1]["x"]
                    c.line(prev_x + (x - prev_x) * 0.5, sub_y, x, sub_y)
                elif i < len(bg) - 1:
                    # 後のノートの方向に寄せる
                    next_x = bg[i + 1]["x"]
                    c.line(x, sub_y, x + (next_x - x) * 0.5, sub_y)
                else:
                    c.line(x, sub_y, x + 5, sub_y)
            else:
                # 連続16分 → 区間全体にサブビーム
                c.line(bg[i]["x"], sub_y, bg[j - 1]["x"], sub_y)
            
            i = j
        else:
            i += 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_renderer.py <musicxml_path> [output.pdf]")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else xml_path.replace(".musicxml", ".pdf")
    
    result = musicxml_to_pdf(xml_path, pdf_path)
    print(f"PDF generated: {result}")

