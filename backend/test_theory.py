"""禁じられた遊びの実データで音楽理論フィルタの効果を確認"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_theory import apply_music_theory_filter, PC_NAMES

# 実際の検出ノート
notes = json.load(open('D:/Music/nextchord-solotab/uploads/20260428-205331-yt-8123dc/notes_assigned.json'))

# コード検出結果を読み込み
chords = json.load(open('D:/Music/nextchord-solotab/uploads/20260428-205331-yt-8123dc/chords.json', encoding='utf-8'))

print(f"Input: {len(notes)} notes, {len(chords)} chords")
print(f"\nFirst 5 chords:")
for c in chords[:5]:
    print(f"  {c.get('start',0):.1f}-{c.get('end',0):.1f}: {c.get('chord','?')}")

# 補正前の冒頭
print(f"\n=== BEFORE (first 18 notes) ===")
for n in notes[:18]:
    pc = PC_NAMES[n['pitch'] % 12]
    print(f"  t={n['start']:.3f} pitch={n['pitch']}({pc}) str={n['string']} fret={n['fret']}")

# 音楽理論フィルタ適用
result = apply_music_theory_filter(notes, chords)

# 補正後の冒頭
print(f"\n=== AFTER (first 18 notes) ===")
for n in result[:18]:
    pc = PC_NAMES[n['pitch'] % 12]
    flag = n.get('_theory_flag', '')
    corr = n.get('_correction_reason', '')
    print(f"  t={n['start']:.3f} pitch={n['pitch']}({pc}) str={n['string']} fret={n['fret']} [{flag}] {corr}")

# 差分サマリ
changed = [n for n in result if n.get('_theory_flag') == 'corrected']
print(f"\n=== Changed notes: {len(changed)} ===")
for n in changed[:20]:
    print(f"  t={n['start']:.3f}: {n['_original_pitch']} -> {n['pitch']} ({n['_correction_reason']})")
