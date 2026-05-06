import io,sys,json
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
sys.path.insert(0,'D:/Music/nextchord-solotab/backend')
sys.path.insert(0,'D:/Music/nextchord-solotab/music-transcription/python')
import os; os.chdir('D:/Music/nextchord-solotab/backend')
from guitar_transcriber import transcribe_guitar
raw = transcribe_guitar('D:/Music/nextchord-solotab/uploads/20260429-003136/converted.wav', onset_threshold=0.5)
notes = [n for n in raw['notes'] if 5.8 < n['start'] < 6.2 and n.get('string', 0) in (5, 6)]
for n in notes:
    print(f"t={n['start']:.3f} p={n['pitch']} s{n['string']}f{n['fret']}")
if not notes:
    print("No bass notes found near M3 start")
    # Check all notes near 5.9-6.1
    all_near = [n for n in raw['notes'] if 5.8 < n['start'] < 6.2]
    for n in all_near:
        print(f"  t={n['start']:.3f} p={n['pitch']} s{n['string']}f{n['fret']}")
