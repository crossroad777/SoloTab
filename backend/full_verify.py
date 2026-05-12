"""Full verification + PNG of E2E GP5."""
import sys
sys.path.insert(0, '.')
import guitarpro as gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = r'D:\Music\nextchord-solotab\uploads\romance_e2e_test\tab.gp5'
song = gp.parse(path)
t = song.tracks[0]
total_measures = len(t.measures)

# --- Verification ---
print(f"=== Full Verification ({total_measures} measures) ===")
ok = 0
for mi in range(total_measures):
    m = t.measures[mi]
    v = m.voices[0]
    nc = sum(1 for b in v.beats if 'rest' not in str(b.status).lower())
    rc = sum(1 for b in v.beats if 'rest' in str(b.status).lower())
    cc = sum(1 for b in v.beats if len(b.notes) > 1 and 'rest' not in str(b.status).lower())
    
    # Check for consecutive same-fret on string 1 (rapid-fire indicator)
    str1_frets = []
    for b in v.beats:
        if 'rest' not in str(b.status).lower():
            for n in b.notes:
                if n.string == 1:
                    str1_frets.append(n.value)
    
    repeats = 0
    for i in range(len(str1_frets) - 1):
        if str1_frets[i] == str1_frets[i+1]:
            repeats += 1
    
    is_ok = nc >= 8 and rc == 0 and cc == 0
    if is_ok: ok += 1
    
    repeat_marker = f" REPEAT={repeats}" if repeats > 0 else ""
    status = "OK" if is_ok else "NG"
    print(f"  M{mi+1:2d}: n={nc} r={rc} c={cc} str1={str1_frets}{repeat_marker} [{status}]")

print(f"\n  Total: {ok}/{total_measures} OK")

# --- PNG ---
measures_per_row = 6
rows = (total_measures + measures_per_row - 1) // measures_per_row
fig, axes = plt.subplots(rows, 1, figsize=(22, rows * 2.5))
if rows == 1: axes = [axes]

for row in range(rows):
    ax = axes[row]
    ax.set_xlim(-0.5, 55)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for s in range(6):
        ax.axhline(y=s, color='#ccc', linewidth=0.5, zorder=0)
    for i, label in enumerate(['e','B','G','D','A','E']):
        ax.text(-0.3, i, label, fontsize=7, va='center', ha='right', fontweight='bold', color='#666')
    
    x_pos = 0.5
    for col in range(measures_per_row):
        mi = row * measures_per_row + col
        if mi >= total_measures: break
        m = t.measures[mi]
        v1 = m.voices[0]
        nc = sum(1 for b in v1.beats if 'rest' not in str(b.status).lower())
        ax.text(x_pos, -0.6, f'{mi+1}', fontsize=6, ha='left', color='#999')
        status_color = '#2d6a4f' if nc >= 9 else ('#e9c46a' if nc >= 8 else '#e76f51')
        ax.text(x_pos, -0.3, f'({nc})', fontsize=5, ha='left', color=status_color)
        
        beat_x = x_pos
        for b in v1.beats:
            if 'rest' in str(b.status).lower():
                beat_x += 0.65
                continue
            for n in b.notes:
                si = n.string - 1
                ax.text(beat_x, si, str(n.value), fontsize=7, ha='center', va='center',
                        fontweight='bold', color='#1a1a2e',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8))
            beat_x += 0.65
        
        if len(m.voices) > 1:
            for b in m.voices[1].beats:
                if 'rest' not in str(b.status).lower():
                    for n in b.notes:
                        si = n.string - 1
                        ax.text(x_pos, si, str(n.value), fontsize=7, ha='center', va='center',
                                fontweight='bold', color='#e63946',
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='#fff0f0', edgecolor='none', alpha=0.8))
        
        x_pos = beat_x + 0.3
        ax.axvline(x=x_pos, color='#888', linewidth=0.8)
        x_pos += 0.3
    ax.axis('off')

fig.suptitle(f'Romance E2E - onset duration ({total_measures} measures) | {song.tempo} BPM | 3/4', fontsize=11, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = r'D:\Music\romance_e2e_verify.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'\nPNG: {out}')
