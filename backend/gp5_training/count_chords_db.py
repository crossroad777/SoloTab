import sys, json
sys.stdout.reconfigure(encoding='utf-8')
with open(r'D:\Music\nextchord-solotab\datasets\chords-db\lib\guitar.json') as f:
    data = json.load(f)

total_positions = 0
for key_chords in data['chords'].values():
    for chord in key_chords:
        total_positions += len(chord.get('positions', []))

keys = data['keys']
suffixes = data['suffixes']
entries = sum(len(v) for v in data['chords'].values())
print(f'Keys: {keys}')
print(f'Suffixes: {len(suffixes)} types')
print(f'Chord entries: {entries}')
print(f'Total voicings (positions with frets+fingers): {total_positions}')
print(f'Suffix list: {suffixes[:20]}...')
