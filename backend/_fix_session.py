"""Copy completed session data to latest failed session for testing"""
import json, shutil, pathlib

uploads = pathlib.Path(r'D:\Music\nextchord-solotab\uploads')
src = uploads / '20260513-234214'
dst = uploads / '20260514-000457'

for f in ['tab.gp5','tab.musicxml','beats.json','notes_assigned.json','converted.wav','techniques.json']:
    sf = src / f
    if sf.exists():
        shutil.copy2(sf, dst / f)
        print("Copied", f, sf.stat().st_size, "bytes")

sj = json.loads((dst/'session.json').read_text(encoding='utf-8'))
src_sj = json.loads((src/'session.json').read_text(encoding='utf-8'))
sj['status'] = 'completed'
sj['error'] = None
sj['bpm'] = src_sj.get('bpm')
sj['total_notes'] = src_sj.get('total_notes')
sj['key'] = src_sj.get('key')
sj['capo'] = src_sj.get('capo', 0)
sj['time_signature'] = src_sj.get('time_signature', '4/4')
sj['result'] = src_sj.get('result', {})
(dst/'session.json').write_text(json.dumps(sj, ensure_ascii=False, indent=2), encoding='utf-8')
print("Updated session to completed, notes:", sj.get('total_notes'))
