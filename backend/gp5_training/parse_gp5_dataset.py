"""
Step 1: GP5 Dataset Parser (v2 - Robust)
==========================================
Parse 17,000+ GP5 files → extract note-level training data
- Skips drums/percussion tracks (fret > 24)
- Per-track try/except for robustness
- Streams output to avoid memory issues
"""
import sys, os, json, time, traceback
from pathlib import Path
from fractions import Fraction


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            return float(obj)
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent.parent))
import guitarpro as gp

# --- Config ---
GP5_DIRS = [
    Path(r"D:\Music\nextchord-solotab\datasets\gprotab_downloads"),
    Path(r"D:\Music\nextchord-solotab\gprotab_downloads"),
]
OUTPUT_DIR = Path(r"D:\Music\nextchord-solotab\backend\gp5_training\data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOTES_FILE = OUTPUT_DIR / "notes_dataset.jsonl"
CHORDS_FILE = OUTPUT_DIR / "chords_dataset.jsonl"
STATS_FILE = OUTPUT_DIR / "dataset_stats.json"

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]


def collect_gp_files():
    extensions = {'.gp3', '.gp4', '.gp5', '.gpx', '.gp'}
    files = []
    for d in GP5_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in extensions and f.is_file():
                files.append(str(f))
    return files


def is_real_guitar_track(track):
    """Filter out drums, vocals, bass, keys — keep only guitar-like tracks"""
    name = (track.name or "").lower()
    n_strings = len(track.strings)
    
    # Exclude bass (4 strings)
    if n_strings < 6:
        return False
    if n_strings > 8:
        return False
    
    # Exclude by name
    exclude_keywords = ['drum', 'perc', 'vocal', 'vox', 'voice', 'sing', 
                        'bass', 'key', 'piano', 'organ', 'synth', 'pad',
                        'string', 'violin', 'cello', 'brass', 'horn',
                        'flute', 'sax', 'trumpet']
    if any(k in name for k in exclude_keywords):
        return False
    
    # Check MIDI channel (channel 10 = drums in GM)
    if hasattr(track, 'channel') and track.channel:
        ch = track.channel
        if hasattr(ch, 'channel') and ch.channel == 9:  # 0-indexed channel 10
            return False
    
    return True


def parse_track_notes(track, song):
    """Parse notes from a single guitar track"""
    notes = []
    chords = []
    
    tuning = [s.value for s in track.strings]
    n_strings = len(tuning)
    tempo = song.tempo
    tick_pos = 0
    ticks_per_beat = 960
    
    for measure_idx, measure in enumerate(track.measures):
        header = song.measureHeaders[measure_idx] if measure_idx < len(song.measureHeaders) else None
        # tempo is global (song.tempo), not per-measure in PyGuitarPro
        
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.status != gp.BeatStatus.normal:
                    tick_pos += getattr(beat.duration, 'time', 960)
                    continue
                
                beat_duration_ticks = getattr(beat.duration, 'time', 960)
                beat_time_sec = tick_pos / ticks_per_beat * (60.0 / max(tempo, 1))
                
                beat_notes = []
                for note in beat.notes:
                    string_num = note.string
                    fret = note.value
                    
                    # Skip invalid
                    if string_num < 1 or string_num > n_strings:
                        continue
                    if fret < 0 or fret > 24:  # drums have fret > 24
                        continue
                    
                    open_pitch = tuning[string_num - 1]
                    pitch = open_pitch + fret
                    
                    # Basic sanity: guitar MIDI range 40-96
                    if pitch < 30 or pitch > 100:
                        continue
                    
                    # Technique detection (safely)
                    techniques = []
                    try:
                        eff = note.effect
                        if eff:
                            if getattr(eff, 'hammer', False): techniques.append("hammer")
                            if getattr(eff, 'slide', None): techniques.append("slide")
                            if getattr(eff, 'bend', None): techniques.append("bend")
                            if getattr(eff, 'harmonic', None): techniques.append("harmonic")
                            if getattr(eff, 'palmMute', False) or getattr(eff, 'isPalmMute', False):
                                techniques.append("palm_mute")
                    except Exception:
                        pass
                    
                    record = {
                        "pitch": pitch,
                        "string": string_num,
                        "fret": fret,
                        "duration_ticks": beat_duration_ticks,
                        "time_sec": round(beat_time_sec, 4),
                        "measure": measure_idx,
                        "tempo": tempo,
                        "tuning": tuning[:6],
                        "n_strings": n_strings,
                        "techniques": techniques,
                        "velocity": note.velocity,
                    }
                    notes.append(record)
                    beat_notes.append(record)
                
                # Chord detection
                if len(beat_notes) >= 2:
                    chords.append({
                        "notes": [{"pitch": n["pitch"], "string": n["string"], "fret": n["fret"]} 
                                  for n in beat_notes],
                        "time_sec": round(beat_time_sec, 4),
                        "measure": measure_idx,
                        "tuning": tuning[:6],
                        "n_strings": n_strings,
                    })
                
                tick_pos += beat_duration_ticks
    
    return notes, chords


def main():
    print("=" * 60)
    print("  Step 1: GP5 Dataset Parser (v2)")
    print("=" * 60)
    
    print("\n[1/4] Collecting GP files...")
    gp_files = collect_gp_files()
    print(f"  Found {len(gp_files)} GP files")
    
    if not gp_files:
        print("  ERROR: No GP files found!")
        return
    
    print(f"\n[2/4] Parsing & streaming to disk...")
    
    notes_f = open(NOTES_FILE, 'w', encoding='utf-8')
    chords_f = open(CHORDS_FILE, 'w', encoding='utf-8')
    
    total_notes = 0
    total_chords = 0
    success = 0
    failed = 0
    tracks_used = 0
    start_time = time.time()
    
    for i, filepath in enumerate(gp_files):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(gp_files) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(gp_files)}] {success} ok, {failed} fail, "
                  f"{total_notes:,} notes, {total_chords:,} chords, "
                  f"{tracks_used} tracks ({rate:.1f} f/s, ETA {eta:.0f}m)")
        
        try:
            song = gp.parse(filepath)
        except UnicodeDecodeError:
            # Try alternate encodings
            song = None
            for enc in ['latin1', 'cp1252', 'utf-8']:
                try:
                    song = gp.parse(filepath, encoding=enc)
                    break
                except Exception:
                    continue
            if song is None:
                failed += 1
                continue
        except Exception as e:
            if failed < 3:
                import traceback
                print(f"  [PARSE ERR] {filepath}: {e}")
                traceback.print_exc()
            failed += 1
            continue
        
        file_had_notes = False
        for track in song.tracks:
            if not is_real_guitar_track(track):
                continue
            
            try:
                notes, chords = parse_track_notes(track, song)
            except Exception as e:
                if tracks_used < 3:
                    import traceback
                    print(f"  [TRACK ERR] {track.name}: {e}")
                    traceback.print_exc()
                continue
            
            if not notes:
                continue
            
            file_had_notes = True
            tracks_used += 1
            
            for note in notes:
                notes_f.write(json.dumps(note, ensure_ascii=False, cls=SafeEncoder) + '\n')
                total_notes += 1
            
            for chord in chords:
                chords_f.write(json.dumps(chord, ensure_ascii=False, cls=SafeEncoder) + '\n')
                total_chords += 1
        
        if file_had_notes:
            success += 1
        else:
            failed += 1
    
    notes_f.close()
    chords_f.close()
    
    elapsed = time.time() - start_time
    print(f"\n  Parsing complete in {elapsed:.0f}s")
    print(f"  Success: {success}, Failed: {failed}")
    print(f"  Tracks used: {tracks_used}")
    print(f"  Notes: {total_notes:,}")
    print(f"  Chords: {total_chords:,}")
    
    # Add context in a second pass (streaming)
    print(f"\n[3/4] Adding context features...")
    
    # Read all notes into memory for context
    all_notes = []
    with open(NOTES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            all_notes.append(json.loads(line))
    
    # Add context
    context_window = 5
    with open(NOTES_FILE, 'w', encoding='utf-8') as f:
        for i, note in enumerate(all_notes):
            note["prev_pitches"] = [all_notes[i-j]["pitch"] for j in range(1, min(context_window+1, i+1))]
            note["prev_strings"] = [all_notes[i-j]["string"] for j in range(1, min(context_window+1, i+1))]
            note["next_pitches"] = [all_notes[i+j]["pitch"] for j in range(1, min(context_window+1, len(all_notes)-i))]
            note["next_strings"] = [all_notes[i+j]["string"] for j in range(1, min(context_window+1, len(all_notes)-i))]
            f.write(json.dumps(note, ensure_ascii=False, cls=SafeEncoder) + '\n')
    
    print(f"  Context added to {len(all_notes):,} notes")
    
    # Stats
    print(f"\n[4/4] Computing statistics...")
    stats = {
        "total_files": len(gp_files),
        "parsed_ok": success,
        "parsed_fail": failed,
        "tracks_used": tracks_used,
        "total_notes": total_notes,
        "total_chords": total_chords,
        "parse_time_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, cls=SafeEncoder)
    
    notes_mb = NOTES_FILE.stat().st_size / 1024 / 1024
    chords_mb = CHORDS_FILE.stat().st_size / 1024 / 1024
    
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"  Files: {success}/{len(gp_files)}")
    print(f"  Notes: {total_notes:,} ({notes_mb:.1f} MB)")
    print(f"  Chords: {total_chords:,} ({chords_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
