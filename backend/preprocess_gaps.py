import os
import sys
import glob
import torch
import numpy as np
import mido

# Add backend directory to path to import string_assigner
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
from string_assigner import assign_strings_dp

# Paths
GAPS_DIR = os.path.join(backend_dir, "..", "datasets", "gaps", "gaps_v1")
AUDIO_DIR = os.path.join(GAPS_DIR, "audio")
MIDI_DIR = os.path.join(GAPS_DIR, "midi")
OUTPUT_DIR = os.path.join(backend_dir, "..", "_processed_gaps_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_midi_to_notes(midi_path):
    mid = mido.MidiFile(midi_path)
    notes = []
    active_notes = {}
    current_time = 0.0
    
    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = {
                'start': current_time,
                'pitch': msg.note,
                'velocity': msg.velocity
            }
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                note = active_notes.pop(msg.note)
                note['end'] = current_time
                notes.append(note)
    
    # Sort by start time
    notes.sort(key=lambda x: x['start'])
    return notes

def process_gaps():
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    processed_count = 0
    
    for wav_file in wav_files:
        basename = os.path.basename(wav_file).replace(".wav", "")
        midi_file = os.path.join(MIDI_DIR, f"{basename}-fine-aligned.mid")
        
        if not os.path.exists(midi_file):
            print(f"MIDI missing: {midi_file}")
            continue
            
        labels_path = os.path.join(OUTPUT_DIR, f"{basename}_labels.pt")
        if os.path.exists(labels_path):
            continue
            
        notes = parse_midi_to_notes(midi_file)
        if not notes:
            continue
            
        # Add DP string assignment
        assigned_notes = assign_strings_dp(notes, max_fret=19)
        
        # Convert to tensor format: [onset, offset, string, fret, pitch]
        # Note: string_assigner returns string 1-6. dataset.py expects string_index 0-5.
        annotation_list = []
        for n in assigned_notes:
            onset = float(n['start'])
            offset = float(n['end'])
            string_idx = int(n['string']) - 1  # 1-6 -> 0-5
            fret = int(n['fret'])
            pitch = float(n['pitch'])
            annotation_list.append((onset, offset, string_idx, fret, pitch))
            
        labels_array = np.array(annotation_list, dtype=np.float32)
        torch.save(torch.from_numpy(labels_array), labels_path)
        processed_count += 1
        print(f"Processed: {basename} ({len(notes)} notes)")
        
    print(f"Finished processing {processed_count} files. Total in dir: {len(glob.glob(os.path.join(OUTPUT_DIR, '*_labels.pt')))}")

if __name__ == "__main__":
    process_gaps()
