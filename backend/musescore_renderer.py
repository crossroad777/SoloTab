"""
musescore_renderer.py — MuseScore 4 を使った五線譜+TAB譜PDF生成
================================================================
SoloTabが生成したMusicXML（TABのみ）を、
五線譜+TAB譜のデュアルスタッフMusicXMLに変換し、
MuseScore 4のCLIでPDFにレンダリングする。

出力は以下の形式:
  - Staff 1: 五線譜（ト音記号、音符・ステム・ビーム・休符の完全な楽譜）
  - Staff 2: TAB譜（6弦のフレット番号）
  - コード記号（Harmony）
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import subprocess
import os
import sys
import tempfile
import shutil
from pathlib import Path

# MuseScore 3 実行ファイルのパス（MuseScore 4はCLIエクスポートが不安定なため3を使用）
MUSESCORE_EXE = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"


def _midi_to_step(midi_num: int) -> str:
    steps = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    return steps[midi_num % 12]

def _midi_to_alter(midi_num: int) -> int:
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    return alters[midi_num % 12]

def _midi_to_octave(midi_num: int) -> int:
    return (midi_num // 12) - 1


def convert_to_dual_staff(input_musicxml: str, output_musicxml: str) -> str:
    """
    TABのみのMusicXMLを、五線譜+TABのデュアルスタッフMusicXMLに変換する。
    
    MusicXMLの構造:
    - <part> に <staves>2</staves> を設定
    - Staff 1: treble clef (五線譜)
    - Staff 2: TAB clef
    - 各 <note> に <staff>1</staff> と <staff>2</staff> の両方を出力
      （実際にはMusicXMLでは1つのnoteに1つのstaffなので、
       backup/forwardを使って2つのスタッフに同じノートを書く）
    """
    tree = ET.parse(input_musicxml)
    root = tree.getroot()
    
    # パートリストを更新
    part_list = root.find("part-list")
    if part_list is not None:
        sp = part_list.find("score-part[@id='P1']")
        if sp is not None:
            sp.find("part-name").text = "Guitar"
    
    part = root.find(".//part[@id='P1']")
    if part is None:
        return input_musicxml
    
    # divisions を取得
    divisions = 12
    div_el = root.find(".//attributes/divisions")
    if div_el is not None and div_el.text:
        divisions = int(div_el.text)
    
    # 拍子情報を取得
    beats_per_bar = 4
    beat_type = 4
    time_el = root.find(".//attributes/time")
    if time_el is not None:
        b = time_el.find("beats")
        bt = time_el.find("beat-type")
        if b is not None and b.text:
            beats_per_bar = int(b.text)
        if bt is not None and bt.text:
            beat_type = int(bt.text)
    
    bar_duration = divisions * beats_per_bar * (4 // beat_type)
    
    # チューニング情報を保存
    tuning_info = []
    staff_details = root.find(".//staff-details")
    if staff_details is not None:
        for st in staff_details.findall("staff-tuning"):
            step = st.find("tuning-step")
            octave = st.find("tuning-octave")
            if step is not None and octave is not None:
                tuning_info.append((st.get("line"), step.text, octave.text))
    
    # 全measureを処理
    for measure_idx, measure in enumerate(part.findall("measure")):
        attrs = measure.find("attributes")
        
        if measure_idx == 0:
            # 最初の小節: attributesを2スタッフ構成に書き換え
            if attrs is None:
                attrs = ET.SubElement(measure, "attributes")
            
            # staves要素を追加
            staves_el = attrs.find("staves")
            if staves_el is None:
                staves_el = ET.SubElement(attrs, "staves")
            staves_el.text = "2"
            
            # 既存のclefとstaff-detailsを削除
            for old_clef in attrs.findall("clef"):
                attrs.remove(old_clef)
            for old_sd in attrs.findall("staff-details"):
                attrs.remove(old_sd)
            
            # Staff 1: ト音記号（五線譜）
            clef1 = ET.SubElement(attrs, "clef", number="1")
            ET.SubElement(clef1, "sign").text = "G"
            ET.SubElement(clef1, "line").text = "2"
            
            # Staff 2: TAB記号
            clef2 = ET.SubElement(attrs, "clef", number="2")
            ET.SubElement(clef2, "sign").text = "TAB"
            ET.SubElement(clef2, "line").text = "5"
            
            # Staff 2 のスタッフ詳細（6弦ギター）
            sd = ET.SubElement(attrs, "staff-details", number="2")
            ET.SubElement(sd, "staff-lines").text = "6"
            if tuning_info:
                for line_num, step, octave in tuning_info:
                    st = ET.SubElement(sd, "staff-tuning", line=line_num)
                    ET.SubElement(st, "tuning-step").text = step
                    ET.SubElement(st, "tuning-octave").text = octave
        
        # ノートとフォワード/バックアップを収集
        # Staff 1（五線譜）のノートを出力した後、
        # backup して Staff 2（TAB）のノートを出力する
        
        original_elements = list(measure)
        
        # ノート以外の要素はそのまま残す
        # ノートには <staff>1</staff> を追加
        # その後 backup + Staff 2 のノートを追加
        
        staff1_notes = []
        staff2_notes = []
        other_elements = []
        total_duration = 0
        
        for elem in original_elements:
            if elem.tag == "note":
                # Staff 1 用のノート: 五線譜用（string/fret情報を除去）
                note1 = _clone_element(elem)
                staff1_el = ET.SubElement(note1, "staff")
                staff1_el.text = "1"
                
                # stem を "down" に（TABでは "none"だったものを修正）
                stem = note1.find("stem")
                if stem is not None:
                    stem.text = "down"
                else:
                    stem_el = ET.SubElement(note1, "stem")
                    stem_el.text = "down"
                
                # 五線譜からtechnical要素を除去（弦番号の丸数字が表示されるのを防ぐ）
                notations = note1.find("notations")
                if notations is not None:
                    for tech in notations.findall("technical"):
                        notations.remove(tech)
                    # notationsが空になったら削除
                    if len(notations) == 0:
                        note1.remove(notations)
                
                # notehead要素も除去（ゴーストノート等のx表記を五線譜では使わない）
                for nh in note1.findall("notehead"):
                    note1.remove(nh)
                
                staff1_notes.append(note1)
                
                # Staff 2 用のノート: TAB用
                note2 = _clone_element(elem)
                staff2_el = ET.SubElement(note2, "staff")
                staff2_el.text = "2"
                
                # stem を "none" に（TABでは不要）
                stem2 = note2.find("stem")
                if stem2 is not None:
                    stem2.text = "none"
                
                staff2_notes.append(note2)
                
                # duration tracking (chord notes don't advance)
                if elem.find("chord") is None:
                    dur_el = elem.find("duration")
                    if dur_el is not None and dur_el.text:
                        total_duration += int(dur_el.text)
            elif elem.tag == "forward":
                dur_el = elem.find("duration")
                dur_val = int(dur_el.text) if dur_el is not None and dur_el.text else 0
                total_duration += dur_val
                
                fwd1 = _clone_element(elem)
                staff1_notes.append(fwd1)
                
                fwd2 = _clone_element(elem)
                staff2_notes.append(fwd2)
            elif elem.tag == "backup":
                dur_el = elem.find("duration")
                dur_val = int(dur_el.text) if dur_el is not None and dur_el.text else 0
                total_duration -= dur_val
                
                bk1 = _clone_element(elem)
                staff1_notes.append(bk1)
                
                bk2 = _clone_element(elem)
                staff2_notes.append(bk2)
            else:
                other_elements.append(elem)
        
        # measure を再構築
        measure.clear()
        measure.set("number", str(measure_idx + 1))
        
        # 非ノート要素を先に追加
        for elem in other_elements:
            measure.append(elem)
        
        # Staff 1 のノート
        for note in staff1_notes:
            measure.append(note)
        
        # Backup (Staff 1 の全duration分戻る)
        if staff2_notes and total_duration > 0:
            backup = ET.SubElement(measure, "backup")
            ET.SubElement(backup, "duration").text = str(total_duration)
        
        # Staff 2 のノート
        for note in staff2_notes:
            measure.append(note)
    
    # 出力
    xml_str = ET.tostring(root, encoding="unicode")
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    header += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    header += '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    
    with open(output_musicxml, "w", encoding="utf-8") as f:
        f.write(header + xml_str)
    
    return output_musicxml


def _clone_element(elem: ET.Element) -> ET.Element:
    """XMLエレメントをディープコピーする"""
    new = ET.Element(elem.tag, elem.attrib)
    new.text = elem.text
    new.tail = elem.tail
    for child in elem:
        new.append(_clone_element(child))
    return new


def render_pdf_with_musescore(musicxml_path: str, output_pdf_path: str, title: str = "") -> str:
    """
    MusicXMLファイルをMuseScore 4でPDFにレンダリングする。
    
    1. TABのみのMusicXMLをデュアルスタッフ（五線譜+TAB）に変換
    2. MuseScore 4 CLI でPDFにエクスポート
    
    Parameters
    ----------
    musicxml_path : str
        入力MusicXMLファイルパス
    output_pdf_path : str  
        出力PDFファイルパス
    
    Returns
    -------
    str
        生成されたPDFのパス
    """
    if not os.path.exists(MUSESCORE_EXE):
        raise FileNotFoundError(f"MuseScore 4 not found at: {MUSESCORE_EXE}")
    
    if not os.path.exists(musicxml_path):
        raise FileNotFoundError(f"MusicXML not found: {musicxml_path}")
    
    # Step 1: デュアルスタッフMusicXMLに変換
    dual_xml = musicxml_path.replace(".musicxml", "_dual.musicxml")
    convert_to_dual_staff(musicxml_path, dual_xml)
    print(f"[MuseScore] Dual-staff MusicXML created: {dual_xml}")
    
    # Step 2: MuseScore CLI でPDFに変換
    cmd = [
        MUSESCORE_EXE,
        "-o", output_pdf_path,
        dual_xml
    ]
    
    print(f"[MuseScore] Rendering PDF: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120,
            encoding="utf-8",
            errors="replace"
        )
        
        if result.returncode != 0:
            try:
                print(f"[MuseScore] stderr: {result.stderr[:500]}")
            except Exception:
                print("[MuseScore] stderr: (encoding error, skipped)")
            # フォールバック: 変換前のMusicXMLで試す
            print("[MuseScore] Trying with original MusicXML...")
            cmd2 = [MUSESCORE_EXE, "-o", output_pdf_path, musicxml_path]
            result2 = subprocess.run(
                cmd2, capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace"
            )
            if result2.returncode != 0:
                raise RuntimeError(f"MuseScore export failed")
        
        print(f"[MuseScore] PDF generated: {output_pdf_path}")
        return output_pdf_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("MuseScore export timed out (120s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python musescore_renderer.py <musicxml_path> [output.pdf]")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else xml_path.replace(".musicxml", "_ms.pdf")
    
    result = render_pdf_with_musescore(xml_path, pdf_path)
    print(f"Done: {result}")
