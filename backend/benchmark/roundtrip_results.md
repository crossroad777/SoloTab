# ラウンドトリップテスト結果 (Recall 上限検証)

**使用合成方式**: Karplus-Strong

### Pattern 1: Single note scale
- Ground Truth ノート数: 15
- 検出ノート数: 13
- Metrics: Precision=0.7692 | Recall=0.6667 | F1=0.7143

### Pattern 2: Chords
- Ground Truth ノート数: 22
- 検出ノート数: 20
- Metrics: Precision=0.9500 | Recall=0.8636 | F1=0.9048

### Pattern 3: High-density fast picking
- Ground Truth ノート数: 32
- 検出ノート数: 17
- Metrics: Precision=0.8235 | Recall=0.4375 | F1=0.5714

### Pattern 4: Arpeggio
- Ground Truth ノート数: 40
- 検出ノート数: 45
- Metrics: Precision=0.6444 | Recall=0.7250 | F1=0.6824

### Pattern 5: 2-voice melody
- Ground Truth ノート数: 16
- 検出ノート数: 16
- Metrics: Precision=0.8125 | Recall=0.8125 | F1=0.8125

## 総合結果 (Overall)
- Total Ground Truth: 125
- Total Detected: 111
- **Overall Recall: 0.6800**
- (GuitarSet Recall: 0.8430)
- 差分: -0.1630

