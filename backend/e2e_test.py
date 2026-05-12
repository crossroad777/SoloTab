"""End-to-end pipeline test with Romance audio."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, '.')

# Find the original audio file
session_dir = Path(r'D:\Music\nextchord-solotab\uploads\20260512-073742')
# Check for wav file
import glob
wav_files = glob.glob(str(session_dir / '*.wav'))
print(f"Session: {session_dir}")
print(f"WAV files: {wav_files}")

# Use existing session data but re-run GP5 generation through the pipeline path
# to verify the beat grid correction triggers
from pipeline import run_pipeline

# Create a new test session
test_session_id = 'romance_e2e_test'
test_session_dir = Path(r'D:\Music\nextchord-solotab\uploads') / test_session_id
test_session_dir.mkdir(parents=True, exist_ok=True)

if wav_files:
    wav_path = Path(wav_files[0])
    print(f"Using: {wav_path}")
    print(f"Test session: {test_session_dir}")
    print()
    
    def progress(step, msg):
        print(f"  [{step}] {msg}")
    
    t0 = time.time()
    try:
        result = run_pipeline(
            session_id=test_session_id,
            session_dir=test_session_dir,
            wav_path=wav_path,
            tuning_name='standard',
            title='Romance E2E Test',
            progress_cb=progress,
            skip_demucs=True,  # Solo guitar, skip separation
        )
        elapsed = time.time() - t0
        print(f"\n--- Pipeline completed in {elapsed:.1f}s ---")
        print(f"BPM: {result['bpm']}")
        print(f"Time Sig: {result['time_signature']}")
        print(f"Notes: {result['total_notes']}")
        print(f"Key: {result['key']}")
        
        # Verify the generated GP5
        gp5_path = test_session_dir / 'tab.gp5'
        if gp5_path.exists():
            import guitarpro as gp
            song = gp.parse(str(gp5_path))
            t = song.tracks[0]
            print(f"\n--- GP5 Verification ---")
            print(f"Measures: {len(t.measures)}")
            
            ok_count = 0
            total = min(8, len(t.measures))
            for mi in range(total):
                m = t.measures[mi]
                v = m.voices[0]
                note_count = sum(1 for b in v.beats if 'rest' not in str(b.status).lower())
                rest_count = sum(1 for b in v.beats if 'rest' in str(b.status).lower())
                chord_count = sum(1 for b in v.beats if len(b.notes) > 1 and 'rest' not in str(b.status).lower())
                ok = note_count >= 8 and rest_count == 0 and chord_count == 0
                if ok:
                    ok_count += 1
                print(f"  M{mi+1}: notes={note_count} rests={rest_count} chords={chord_count} -> {'OK' if ok else 'NG'}")
            print(f"\n  First {total} measures: {ok_count}/{total} OK")
        else:
            print("GP5 file not found!")
            
    except Exception as e:
        import traceback
        print(f"\n--- Pipeline FAILED ---")
        traceback.print_exc()
else:
    print("No WAV file found in session directory")
    # List what's there
    print("Files:", list(session_dir.iterdir()))
