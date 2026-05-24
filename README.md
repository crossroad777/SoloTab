# SoloTab

**Audio → Guitar Tab + Fingering — Fully Automatic**

音声ファイルからギタータブ譜と運指を自動生成するシステム。

---

## Features / 機能

| Feature | Description |
|---------|-------------|
| 🎵 **Audio → MIDI** | Pitch detection via deep learning (Basic Pitch) |
| 🎸 **MIDI → Tab** | String assignment optimized for playability |
| 🤚 **Tab → Fingering** | CNN + Viterbi DP hybrid (79.9% accuracy, 33K notes) |
| 🎼 **Technique Detection** | Hammer-on, pull-off, slide, bend, vibrato |
| 📄 **GP5 Export** | Guitar Pro format output |

## Architecture / アーキテクチャ

```
Audio (.mp3/.wav)
    ↓ Basic Pitch (ML)
MIDI notes
    ↓ String Assigner (DP)
Tab (string + fret)
    ↓ Technique Detector (CNN)
Techniques (H/P/Slide/Bend)
    ↓ Finger Assigner (CNN + Viterbi DP)
Fingering (1-4)
    ↓ GP Renderer
Guitar Pro 5 (.gp5)
```

## Fingering System / 運指システム

CNN×Viterbi DPの共同最適化による自動運指。

### Accuracy / 精度

| Metric | Value |
|--------|-------|
| **Total** | **79.9%** (26,673 / 33,378 notes) |
| GP5 Solo | 84.4% (15,724 / 18,623) |
| Chord | 74.2% (10,949 / 14,755) |
| CNN Standalone | 99.5% |
| ±1 Finger Tolerance | 92.9% |
| Regression Tests | 18/18 |

### Optimization History / 最適化履歴

| Phase | Total | CNN | Key Change |
|-------|-------|-----|------------|
| Baseline | ~45% | ~45% | Rule-based only |
| +Phase 3 | 64.3% | ~45% | Optuna weight tuning |
| +CNN FT | 67.2% | ~74% | First CNN fine-tuning |
| +Phase 4 | 77.3% | 95.4% | CNN-Viterbi co-optimization |
| +Phase 5 | 79.2% | 97.6% | Error-aware training |
| **+Phase 6** | **79.9%** | **99.5%** | **ctx=5+7 ensemble, final** |

### Key Findings / 主要な発見

- **Context window**: ctx=7 is optimal (bell curve: 3→5→**7**→15 = 96.7→99.0→**99.5**→96.5%)
- **CNN trust scales with accuracy**: w_cnn_prior: 4.26 → 21.07 → 29.19 → **34.99**
- **CNN >> LSTM >> Transformer**: 99.5% vs 77.9% vs 73.1% — local context wins
- **Convergence**: +10.1% → +1.9% → +0.4% → -0.2% (saturated)

## Quick Start / クイックスタート

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

## Tech Stack / 技術スタック

- **Backend**: Python, PyTorch, FastAPI
- **Frontend**: React (Vite)
- **ML Models**: CNN (finger prediction), Basic Pitch (audio→MIDI)
- **Optimization**: Optuna (Bayesian hyperparameter search)

## Docs / ドキュメント

- [運指アルゴリズム論文 (JA)](docs/fingering_rules_paper_ja.md)
- [Fingering Algorithm Paper (EN)](docs/fingering_rules_paper_en.md)

## License / ライセンス

MIT
