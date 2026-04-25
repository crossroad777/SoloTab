"""IDMT-SMT-GUITAR_V2 XML→JAMS変換スクリプト"""
import os, glob
import xml.etree.ElementTree as ET
import jams
import librosa
from pathlib import Path

BASE = r'd:\Music\nextchord-solotab\datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2'
OUT = r'd:\Music\nextchord-solotab\datasets\IDMT-SMT-V2_jams'

def convert_xml_to_jams(xml_path, wav_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    jam = jams.JAMS()
    jam.file_metadata.title = Path(wav_path).stem
    try:
        jam.file_metadata.duration = librosa.get_duration(path=wav_path)
    except:
        jam.file_metadata.duration = 10.0
    
    ann = jams.Annotation(namespace='note_midi')
    ann.annotation_metadata.corpus = 'IDMT-SMT-GUITAR_V2'
    
    for event in root.iter('event'):
        pitch = event.findtext('pitch')
        onset = event.findtext('onsetSec')
        offset = event.findtext('offsetSec')
        fret = event.findtext('fretNumber')
        string = event.findtext('stringNumber')
        
        if pitch and onset:
            try:
                t = float(onset)
                p = int(pitch)
                dur = float(offset) - t if offset else 0.5
                ann.append(time=t, duration=max(dur, 0.01), value=p, confidence=1.0)
            except (ValueError, TypeError):
                pass
    
    if len(ann) > 0:
        jam.annotations.append(ann)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        jam.save(out_path)
        return True
    return False

def main():
    xmls = glob.glob(os.path.join(BASE, '**', '*.xml'), recursive=True)
    wavs = glob.glob(os.path.join(BASE, '**', '*.wav'), recursive=True)
    print(f'XMLs: {len(xmls)}, WAVs: {len(wavs)}')
    
    wav_map = {Path(w).stem: w for w in wavs}
    
    converted = 0
    for xml_path in xmls:
        # XMLからWAVファイル名を取得
        try:
            tree = ET.parse(xml_path)
            wav_name = tree.find('.//audioFileName')
            if wav_name is not None:
                stem = Path(wav_name.text).stem
            else:
                stem = Path(xml_path).stem
        except:
            stem = Path(xml_path).stem
        
        wav_path = wav_map.get(stem)
        if wav_path:
            out = os.path.join(OUT, stem + '.jams')
            if convert_xml_to_jams(xml_path, wav_path, out):
                converted += 1
    
    print(f'Converted: {converted} XML→JAMS')

if __name__ == '__main__':
    main()
