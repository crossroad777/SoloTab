"""
GP5コーパス（4334ファイル）の指番号データ調査
gprotab_downloads/から全ファイルをスキャン
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
import guitarpro

GP5_DIR = r"D:\Music\nextchord-solotab\gprotab_downloads"

def scan_all_gp5():
    files = []
    for root, dirs, fnames in os.walk(GP5_DIR):
        for f in fnames:
            if f.lower().endswith(('.gp5', '.gp4', '.gp3', '.gp')):
                files.append(os.path.join(root, f))
    return files

def analyze(files):
    stats = {
        'total': 0, 'ok': 0, 'err': 0,
        'files_lh': 0, 'files_rh': 0,
        'notes_total': 0, 'notes_lh': 0, 'notes_rh': 0,
        'lh_dist': {}, 'rh_dist': {},
        'examples_lh': [],
    }
    t0 = time.time()
    
    for fp in files:
        stats['total'] += 1
        try:
            song = guitarpro.parse(fp)
        except:
            stats['err'] += 1
            continue
        stats['ok'] += 1
        
        file_lh = False
        file_rh = False
        
        for track in song.tracks:
            for measure in track.measures:
                for voice in measure.voices:
                    for beat in voice.beats:
                        for note in beat.notes:
                            stats['notes_total'] += 1
                            lh = getattr(note.effect, 'leftHandFinger', None)
                            if lh is not None and lh.value >= 0:
                                stats['notes_lh'] += 1
                                k = lh.name
                                stats['lh_dist'][k] = stats['lh_dist'].get(k, 0) + 1
                                file_lh = True
                            rh = getattr(note.effect, 'rightHandFinger', None)
                            if rh is not None and rh.value >= 0:
                                stats['notes_rh'] += 1
                                k = rh.name
                                stats['rh_dist'][k] = stats['rh_dist'].get(k, 0) + 1
                                file_rh = True
        
        if file_lh:
            stats['files_lh'] += 1
            if len(stats['examples_lh']) < 10:
                stats['examples_lh'].append(os.path.basename(fp))
        if file_rh:
            stats['files_rh'] += 1
        
        if stats['total'] % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ...{stats['total']}/{len(files)} ({elapsed:.0f}s) LH={stats['files_lh']} RH={stats['files_rh']}")
    
    return stats

print("=" * 60)
print("  GP5 指番号データ調査 (4334ファイル)")
print("=" * 60)

files = scan_all_gp5()
print(f"\n  {len(files)} GP5 files found")
print(f"\n  分析中...")

stats = analyze(files)
elapsed = time.time() - time.time()

print(f"\n{'=' * 60}")
print(f"  結果")
print(f"{'=' * 60}")
print(f"  パース成功: {stats['ok']}/{stats['total']} ({stats['err']} errors)")
print(f"  総ノート数: {stats['notes_total']:,}")
print(f"")
print(f"  左手指番号: {stats['files_lh']} files ({stats['notes_lh']:,} notes)")
print(f"  右手指番号: {stats['files_rh']} files ({stats['notes_rh']:,} notes)")
if stats['lh_dist']:
    print(f"\n  左手指番号分布:")
    for k, v in sorted(stats['lh_dist'].items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:,}")
if stats['rh_dist']:
    print(f"\n  右手指番号分布:")
    for k, v in sorted(stats['rh_dist'].items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:,}")
if stats['examples_lh']:
    print(f"\n  指番号付きファイル例:")
    for f in stats['examples_lh']:
        print(f"    {f}")

# 結果をJSONに保存
with open(r'D:\Music\nextchord-solotab\gp_training_data\fingering_annotation_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"\n  結果を fingering_annotation_stats.json に保存")
