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
SYSTEM_SPACING = 32  # 段間の余白(pt)
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
    c.drawCentredString(PAGE_W / 2, y, title)
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
    
    for measure_el in part.findall("measure"):
        measure_num = measure_el.get("number", "?")
        notes = []
        chord_name = None
        
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
            
            # テクニック情報
            technique = "normal"
            harmonic = technical.find("harmonic")
            if harmonic is not None:
                technique = "harmonic"
            
            notehead = note_el.find("notehead")
            if notehead is not None and notehead.text == "x":
                technique = "ghost"
            
            notes.append({
                "string": string_num,
                "fret": fret_num,
                "chord": is_chord,
                "technique": technique,
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
    """1小節内のノートを描画"""
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
    
    # 各グループを均等配置
    num_groups = len(groups)
    padding = 10  # 小節端からの余白
    usable_width = bar_width - padding * 2
    
    for g_idx, group in enumerate(groups):
        if num_groups == 1:
            note_x = bar_x + bar_width / 2
        else:
            note_x = bar_x + padding + (g_idx / (num_groups)) * usable_width + usable_width / (num_groups * 2)
        
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
            if note["technique"] == "harmonic":
                c.setFillColorRGB(0, 0.5, 0)
            elif note["technique"] == "ghost":
                c.setFillColorRGB(0.5, 0.5, 0.5)
            else:
                c.setFillColorRGB(0, 0, 0)
            c.drawCentredString(note_x, note_y - 3, fret_text)
            c.setFillColorRGB(0, 0, 0)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_renderer.py <musicxml_path> [output.pdf]")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else xml_path.replace(".musicxml", ".pdf")
    
    result = musicxml_to_pdf(xml_path, pdf_path)
    print(f"PDF generated: {result}")
