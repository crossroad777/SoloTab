"""Fine grid search: w_ts=0.1-0.8, w_tf=0-0.03"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Reuse benchmark_cnn_viterbi
from benchmark_cnn_viterbi import load_solo_data, evaluate

data = load_solo_data(max_files=60)
print(f"Files: {len(data)}")

best_acc = 0; best_cfg = ""

print(f"{'w_ts':>6s} {'w_tf':>6s} {'Acc':>8s}")
print("-" * 25)

for wts_10 in range(1, 9):  # 0.1 to 0.8
    wts = wts_10 / 10.0
    for wtf_100 in [0, 1, 2, 3]:  # 0, 0.01, 0.02, 0.03
        wtf = wtf_100 / 100.0
        c, t, ps, pst = evaluate(data, wts, wtf)
        acc = c / t * 100 if t > 0 else 0
        marker = " *" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_cfg = f"w_ts={wts}, w_tf={wtf}"
            best_ps = ps; best_pst = pst
        print(f"{wts:6.1f} {wtf:6.2f} {acc:7.1f}%{marker}")

print(f"\nBest: {best_cfg} -> {best_acc:.1f}%")
for s in range(1, 7):
    sn = ["E4", "B3", "G3", "D3", "A2", "E2"][s - 1]
    c2 = best_ps.get(s, 0); t2 = best_pst.get(s, 0)
    a2 = c2 / t2 * 100 if t2 > 0 else 0
    print(f"  S{s}({sn}): {c2}/{t2} = {a2:.1f}%")
