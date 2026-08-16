import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(r'D:\Music\chordlink-solotab\backend').resolve()))
from gp_renderer import notes_to_gp5
import guitarpro

def build_phrase_minor_pentatonic():
    """1. マイナー・ペンタトニック (プレーン/巻弦, 2-3フレット間隔)"""
    return [
        (6, 5, False), (6, 8, True),   # A(6弦5) -> C(6弦8) H
        (5, 5, False), (5, 7, True),   # D -> E H
        (4, 5, False), (4, 7, True),   # G -> A H
        (3, 5, False), (3, 7, True),   # C -> D H
        (2, 5, False), (2, 8, True),   # E -> G H
        (1, 5, False), (1, 8, True),   # A -> C H
        
        # Descending (Pull-offs)
        (1, 8, False), (1, 5, True),   # P
        (2, 8, False), (2, 5, True),   # P
        (3, 7, False), (3, 5, True),   # P
        (4, 7, False), (4, 5, True),   # P
        (5, 7, False), (5, 5, True),   # P
        (6, 8, False), (6, 5, True),   # P
    ]

def build_phrase_3nps():
    """2. 3ノート・パー・ストリング (弦移動, フュージョン的アプローチ)"""
    return [
        (6, 5, False), (6, 7, True), (6, 8, True), # A B C (H, H)
        (5, 5, False), (5, 7, True), (5, 8, True), # D E F (H, H)
        (4, 5, False), (4, 7, True), (4, 9, True), # G A B (H, H)
        (3, 5, False), (3, 7, True), (3, 9, True), # C D E (H, H)
        
        # Descending
        (3, 9, False), (3, 7, True), (3, 5, True), # P P
        (4, 9, False), (4, 7, True), (4, 5, True), # P P
        (5, 8, False), (5, 7, True), (5, 5, True), # P P
        (6, 8, False), (6, 7, True), (6, 5, True), # P P
    ]

def build_phrase_chromatic():
    """3. クロマチック (半音階, 1フレット間隔 - アタック極弱)"""
    return [
        (3, 5, False), (3, 6, True), (3, 7, True), (3, 8, True),
        (2, 5, False), (2, 6, True), (2, 7, True), (2, 8, True),
        
        # Descending
        (2, 8, False), (2, 7, True), (2, 6, True), (2, 5, True),
        (3, 8, False), (3, 7, True), (3, 6, True), (3, 5, True),
    ]

def build_phrase_stretch():
    """4. 広域ストレッチ (3-4フレット間隔)"""
    return [
        (4, 2, False), (4, 6, True),  # 4 frets H
        (3, 2, False), (3, 6, True),  # 4 frets H
        (2, 2, False), (2, 7, True),  # 5 frets H
        (1, 2, False), (1, 7, True),  # 5 frets H
        
        (1, 7, False), (1, 2, True),  # P
        (2, 7, False), (2, 2, True),  # P
        (3, 6, False), (3, 2, True),  # P
        (4, 6, False), (4, 2, True),  # P
    ]

def generate_gp5(bpm, filename):
    phrases = [
        build_phrase_minor_pentatonic(),
        build_phrase_3nps(),
        build_phrase_chromatic(),
        build_phrase_stretch()
    ]
    
    notes = []
    current_time = 0.0
    # 8th note duration = (60 / bpm) / 2
    note_dur = (60.0 / bpm) / 2.0
    
    # --- Pattern A: H/P ---
    for phrase in phrases:
        for (string, fret, is_legato) in phrase:
            technique = "normal"
            if is_legato:
                # If current fret is higher than previous fret -> Hammer-on
                # If lower -> Pull-off
                technique = "h" if notes[-1]["fret"] < fret else "p"
                
            notes.append({
                "start": current_time,
                "end": current_time + note_dur - 0.01,
                "pitch": 60,
                "string": string,
                "fret": fret,
                "technique": technique
            })
            current_time += note_dur
        # Rest for 2 beats
        current_time += (60.0 / bpm) * 2.0
        
    # Rest for 1 measure (4 beats) before Pattern B
    current_time += (60.0 / bpm) * 4.0
    
    # --- Pattern B: All Picked ---
    for phrase in phrases:
        for (string, fret, _) in phrase:
            notes.append({
                "start": current_time,
                "end": current_time + note_dur - 0.01,
                "pitch": 60,
                "string": string,
                "fret": fret,
                "technique": "normal"
            })
            current_time += note_dur
        # Rest for 2 beats
        current_time += (60.0 / bpm) * 2.0
        
    # Generate beats array
    total_beats = int(current_time / (60.0 / bpm)) + 4
    beats = [i * (60.0 / bpm) for i in range(total_beats)]
    
    song_bytes = notes_to_gp5(notes, beats=beats, bpm=bpm)
    out_path = Path(__file__).parent.parent / "docs" / filename
    out_path.write_bytes(song_bytes)
    print(f"Generated: {out_path.absolute()} (BPM: {bpm})")

if __name__ == "__main__":
    generate_gp5(80, "HP_Test_Slow_80BPM.gp5")
    generate_gp5(110, "HP_Test_Mid_110BPM.gp5")
    generate_gp5(150, "HP_Test_Fast_150BPM.gp5")
