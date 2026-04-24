import torch
import os
import sys

# Original Raw loading
raw_pt_file = [f for f in os.listdir('_processed_gaps_data') if f.endswith('.pt')][0]
original_data = torch.load(os.path.join('_processed_gaps_data', raw_pt_file), map_location='cpu', weights_only=False)

# After GAPSAcousticDataset transformation
sys.path.insert(0, 'backend')
from train_acoustic_mixed import GAPSAcousticDataset
ds = GAPSAcousticDataset('_processed_gaps_data', 'datasets/gaps/gaps_v1/audio')

# Fetch the first sample's raw_labels representation output by the loader
# It returns: features, (onset, fret), raw_labels, track_id
transformed_labels = ds[0][2]

output = []
output.append("【GAPSデータ・弦番号の反転検証（先頭10ノート）】")
output.append("修正前 (元データ) -> 修正後 (学習スクリプト内)")
output.append("-" * 60)

for i in range(min(10, len(transformed_labels))):
    old_str = original_data[i, 2].item()
    new_str = transformed_labels[i, 2].item()
    pitch = transformed_labels[i, 4].item()
    fret = transformed_labels[i, 3].item()
    
    old_meaning = "High E (1弦)" if old_str == 0 else "Low E (6弦)" if old_str == 5 else f"弦 {old_str}"
    new_meaning = "High E (1弦)" if new_str == 5 else "Low E (6弦)" if new_str == 0 else f"弦 {new_str}"
    
    # Just show the raw numbers
    output.append(f"ノート{i+1} [Pitch {pitch} / Fret {fret}]")
    output.append(f"  └ 旧・弦インデックス: {old_str} -> 新・弦インデックス: {new_str}")

with open("gaps_test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
