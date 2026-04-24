"""
AG-PT-set CSV→JAMS変換スクリプト
AG-PT-setのピッチラベルCSVをJAMSフォーマットに変換する。
CSVフォーマット: onset_time_seconds, midi_pitch, frequency_aubiopitch
"""
import os, csv, glob
import jams
import numpy as np
from pathlib import Path

BASE_DIR = r'd:\Music\nextchord-solotab\datasets\AG-PT-set\aGPTset'
OUTPUT_DIR = r'd:\Music\nextchord-solotab\datasets\AG-PT-set_jams'

# ギター標準チューニング
TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4

def midi_to_string_fret(midi_pitch):
    """MIDIピッチから最適な弦・フレットを推定"""
    best_string, best_fret = 1, 0
    best_diff = 999
    for s_idx, open_pitch in enumerate(TUNING):
        fret = midi_pitch - open_pitch
        if 0 <= fret <= 24:
            # なるべく低いフレットを優先
            if fret < best_diff:
                best_diff = fret
                best_string = s_idx + 1
                best_fret = fret
    return best_string, best_fret

def convert_csv_to_jams(csv_path, wav_path, output_jams_path):
    """1つのCSV+WAVペアをJAMSに変換"""
    import librosa
    jam = jams.JAMS()
    
    # ファイル情報
    jam.file_metadata.title = Path(wav_path).stem
    try:
        duration = librosa.get_duration(path=wav_path)
    except Exception:
        duration = 30.0
    jam.file_metadata.duration = duration
    
    # CSVを読む
    notes = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    onset = float(row[0].strip())
                    pitch = int(float(row[1].strip()))
                    if 40 <= pitch <= 88:  # ギター音域
                        notes.append((onset, pitch))
                except (ValueError, IndexError):
                    continue
    
    if not notes:
        return False
    
    # 弦ごとにアノテーション
    ann = jams.Annotation(namespace='note_midi')
    ann.annotation_metadata.corpus = 'AG-PT-set'
    
    for onset, pitch in sorted(notes):
        string, fret = midi_to_string_fret(pitch)
        duration = 0.5  # デフォルト持続時間
        ann.append(time=onset, duration=duration, value=pitch,
                   confidence=1.0)
    
    jam.annotations.append(ann)
    
    os.makedirs(os.path.dirname(output_jams_path), exist_ok=True)
    jam.save(output_jams_path)
    return True

def main():
    data_dir = os.path.join(BASE_DIR, 'data')
    
    # WAVファイルを探す
    wav_files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    print(f'Found {len(wav_files)} WAV files')
    
    # CSVファイルを探す
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    print(f'Found {len(csv_files)} CSV files')
    
    # CSVのベース名からWAVへのマッピング
    csv_base = {}
    for c in csv_files:
        # CSVの名前からWAV名を推測
        name = Path(c).stem.replace('_pitchlabel', '')
        csv_base[name] = c
    
    converted = 0
    for wav in wav_files:
        wav_stem = Path(wav).stem
        csv_path = csv_base.get(wav_stem)
        if csv_path:
            out_jams = os.path.join(OUTPUT_DIR, wav_stem + '.jams')
            if convert_csv_to_jams(csv_path, wav, out_jams):
                converted += 1
    
    print(f'\nConverted: {converted} CSV→JAMS pairs')
    print(f'Output: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
