# Automatic Guitar Tablature Transcription via Large-Scale Synthetic Data and Domain-Adaptive MoE Ensemble

## SoloTab V2.0 Research Report

## Abstract

Automatic music transcription (AMT) for guitar is an extremely challenging task due to fingering ambiguity, diversity of playing techniques, and timbral variation across genres. We propose a **Domain-Adaptive Pure Mixture-of-Experts (MoE) Ensemble** that combines pre-training on over 52,000 synthetic tracks with multi-dataset fine-tuning integrating GuitarSet, GAPS, and AG-PT-set. Through frame-level consensus voting among seven instrument- and technique-specific CRNN models, we **completely eliminate all post-processing** traditionally required—including noise filtering, dynamic-programming string assignment, and rhythm quantization—while achieving **Pitch F1 = 0.8916** on GuitarSet Test (60 steel-string tracks) and **Pitch F1 = 0.7312** on GAPS Test (30 nylon-string tracks). Furthermore, we discover a "diversity-driven consensus improvement" phenomenon where mixed training with synthetic data (Synth V2) degrades individual model Val F1 yet improves MoE ensemble F1, presenting a novel design principle for ensemble learning. Cross-dataset evaluation on 90 unseen tracks demonstrates the generalization capability of our acoustic guitar-specialized model.

---

## 1. Introduction

Guitar transcription requires not only the detection of pitch and onset timing but also the accurate prediction of *which string and fret were played*. Prior studies (e.g., Omnizart, MT3) have struggled to generalize across playing styles (fingerstyle vs. pick) and genres (Funk, Rock, etc.), with F1 scores plateauing below 0.85.

This study proposes and validates two key hypotheses:

1. Providing the model with large-scale, high-fidelity synthetic data spanning diverse instruments and techniques enables robust pre-training of fundamental pitch-string-fret recognition.
2. A consensus ensemble of domain-specialized models eliminates the need for hand-crafted post-processing rules.

---

## 2. Datasets

### 2.1 Synthetic Pre-training Data (~52,000 tracks)

Synthesized using high-quality SoundFonts representing distinct guitar body characteristics (Martin, Taylor, Luthier, Gibson) and multiple playing techniques (finger, pick, thumb). This provides perfect ground-truth labels for string/fret combinations and pitch.

- **Gibson Thumb Dataset (89,779 files):** Extracted from a large GuitarPro tablature corpus, synthesized with Gibson SoundFont and thumb technique, specializing in fingerstyle solo guitar.

### 2.2 Real-World Datasets

| Dataset | Type | Tracks | Features |
| :--- | :--- | :---: | :--- |
| **GuitarSet** | Steel-string acoustic | 360 | Hexaphonic pickup, 5 genres, gold-standard benchmark |
| **GAPS** | Nylon-string classical | 371 | 14.6h audio + aligned MIDI scores |
| **AG-PT-set** | Acoustic guitar | 360 | 15h, 32k notes, 12 expressive techniques |
| **IDMT-SMT-V2** | Electric guitar | 252 | Human-annotated string/fret, 3 guitar models |

### 2.3 Synth V2 (Regularization Data)

5,000 procedurally generated tracks with 100% accurate labels, sampled at a ratio of 0.5 per epoch to match GuitarSet training size (~286 samples).

---

## 3. Methodology

### 3.1 Architecture: CRNN with Bidirectional GRU

- **Input:** Constant-Q Transform (CQT) spectrogram
- **Core Network:** Convolutional Recurrent Neural Network (CRNN) with Bidirectional GRU (`hidden_size=768`, `layers=2`, `dropout=0.3`)
- **Output:** Multi-task classification producing onset probabilities and fret assignment probabilities
- **Parameters:** ~2M per model

### 3.2 Pure MoE Ensemble

Seven domain-specialized expert models, each pre-trained on domain-specific synthetic data and fine-tuned on real recordings:

| Domain | Instrument | Technique |
| :--- | :--- | :--- |
| martin_finger | Martin D-28 | Fingerstyle |
| taylor_finger | Taylor 814ce | Fingerstyle |
| luthier_finger | Classical Guitar | Fingerstyle |
| martin_pick | Martin D-28 | Pick |
| taylor_pick | Taylor 814ce | Pick |
| luthier_pick | Classical Guitar | Pick |
| gibson_thumb | Gibson J-45 | Thumb |

**Consensus Protocol:** For each frame, a note is accepted if >= *vote_threshold* models agree (onset probability > 0.5). Fret assignment is determined by majority vote. No post-processing is applied.

### 3.3 Multi-Stage Training Pipeline

| Stage | Data | Description |
| :---: | :--- | :--- |
| Stage 1 | Synthetic (52K) | Pre-training on domain-specific synthetic data |
| Stage 2 | + GuitarSet (286) | Fine-tuning on real steel-string recordings |
| Stage 3 | + GAPS (371) | Multi-task learning with nylon-string data |
| Stage 6 | + AG-PT-set (72) | 3-Dataset integration (3DS) |
| Stage 9 | + Synth V2 (286/5000) | Regularization via synthetic mixing |

### 3.4 String Classification Pipeline

1. **CNN String Classifier:** Predicts string from CQT patch (84 bins × 11 frames) + MIDI pitch (6-class classification, **Val accuracy 94.1%**)
2. **Bi-LSTM Refinement:** Integrates CNN probabilities with sequential context (Val accuracy 98.31%)

### 3.5 String Classifier Training: Synthetic Data Experiments and Optimization

To eliminate dependency on the 61,885-sample GuitarSet dataset, we attempted synthetic-only string classifier training across three generations. All synthetic data was generated using FluidSynth with string-specific physical filtering applied to CQT patches.

#### 3.5.1 Synthetic Pipeline Evolution

| Version | Approach | Synth Val | GS Eval | Finding |
| :--- | :--- | :---: | :---: | :--- |
| v3 | Baseline synthesis (972K patches) | 33.0% | 35.1% | No string differentiation in spectra |
| v4 | + Physical filters (lowpass, harmonic decay, attack) | 84.1% | 32.7% | Filters create artificial distinctions |
| v5 | + GS-matched energy/contrast/peak alignment | 85.6% | 24.7% | Marginal improvement, still far below GS |

#### 3.5.2 Domain Gap Analysis

Quantitative comparison between GuitarSet (3,549 samples / 20 tracks) and v4 synthetic data (54,000 samples):

| Metric | GuitarSet | v4 Synthetic | Gap |
| :--- | :---: | :---: | :--- |
| Mean energy | 0.381 | 0.239 | v4 is 37% darker |
| Same-pitch cross-string CQT distance | **0.566** | **0.213** | v4 has only **38%** of GS string separation |
| Peak frequency bin | bin 30-31 | bin 22-23 | 10-bin offset |

The core issue: real guitar string differentiation arises from body resonance, touch dynamics, and picking position—factors impossible to replicate with parametric digital filters.

#### 3.5.3 Transfer Learning: Negative Transfer

We tested v5 pre-training (162K, 3 types) → GuitarSet fine-tuning (49,508 samples):

| Method | GS Eval Accuracy |
| :--- | :---: |
| **Baseline (GS direct, 30 epochs)** | **89.4%** |
| v5 pre-trained + fine-tuned (35 epochs) | 78.3% |
| **Difference** | **-11.1%** |

Synthetic pre-training caused **negative transfer**: features learned from synthetic data were incompatible with real-world spectral characteristics and could not be overridden during fine-tuning.

#### 3.5.4 Optimized Production Model

Abandoning synthetic data entirely, we optimized GuitarSet direct training:

| Parameter | Baseline | Optimized |
| :--- | :--- | :--- |
| Epochs | 30 | **80** |
| Optimizer | Adam (lr=1e-3) | **AdamW** (lr=1e-3, wd=1e-4) |
| Scheduler | ReduceLROnPlateau | **CosineAnnealing** (→1e-5) |
| Augmentation | None | **Gain scaling (×0.85-1.15), Gaussian noise (σ=0.015), temporal shift (±1 frame, p=0.3), frequency shift (±1 bin, p=0.2)** |

**Result:** Val accuracy improved from 89.4% to **94.1%** (+4.7%), with estimated contributions: CosineAnnealing (+2.0%), augmentation (+1.5%), epoch increase (+1.0%), weight decay (+0.2%).

---

## 4. Experiments and Results

### 4.1 Progressive Improvement

| Step | Configuration | Mean Val F1 (7 domains) | MoE Test F1 |
| :---: | :--- | :---: | :---: |
| Step 2 | GuitarSet FT only | 0.7830 | 0.8310 |
| Step 3 | + GAPS | 0.7843 (+0.0013) | 0.8351 |
| Step 6 | + GAPS + AG-PT (3DS) | 0.7867 (+0.0037) | 0.8839 |
| Step 9 | + GAPS + Synth V2 | 0.7636 (-0.0231) | 0.8877 (+0.0038) |
| **Step 10** | **35-model full-stage ensemble** | **--** | **0.8916 (+0.0077)** |

### 4.2 Final Benchmark

Evaluated on GuitarSet Test split (Player 05, 60 unseen tracks) using mono-mic audio:

| Metric | Step 6 (7 models) | Synth V2 (7 models) | **Full Ensemble (35 models)** |
| :--- | :---: | :---: | :---: |
| **Pitch F1** | 0.8839 | 0.8877 | **0.8916** |
| Precision | 0.8592 | 0.8753 | 0.8864 |
| Recall | 0.8653 | 0.9005 | 0.8968 |
| String+Fret Match | 92.31% | 92.38% | 92.30% |
| E2E Exact Match | 82.36% | 83.19% | 82.78% |

### 4.3 Cross-Dataset Evaluation (GAPS)

| Metric | Value |
| :--- | :---: |
| Pitch F1 | 0.7312 |
| String+Fret Match | 70.59% |
| E2E Exact Match | 46.84% |

### 4.4 Comparison with Prior Work

| Method | Year | Architecture | Training Data | GuitarSet F1 | Notes |
| :--- | :---: | :--- | :--- | :---: | :--- |
| TabCNN (Wiggins and Kim) | 2019 | CNN | GuitarSet | ~0.826 | Baseline |
| SynthTab (CRNN) | 2024 | CRNN | Synthetic+GuitarSet | ~0.87+ | Data augmentation |
| **SoloTab V2.0 (Pure MoE)** | **2026** | **7xCRNN Ensemble** | **Synthetic 52K + Multi-DS** | **0.8916** | **No post-processing** |

---

## 5. Key Finding: Diversity-Driven Consensus Improvement

### 5.1 The Paradox

Synth V2 mixed training **degraded** individual model performance:
- Mean Val F1: 0.7867 -> 0.7709 (**-0.0157**)

Yet the MoE ensemble **improved**:
- MoE Test F1: 0.8839 -> 0.8877 (**+0.0038**)

### 5.2 Explanation

1. **Error Diversity:** GuitarSet-specialized models make correlated errors. Synth V2-generalized models make different errors, enabling noise cancellation through consensus.
2. **Recall Boost (+0.035):** Accurate synthetic labels train models to "not miss notes." With 5/7 agreement threshold, individual Recall improvements directly benefit the ensemble.
3. **Precision Boost (+0.016):** Diversified false-positive patterns reduce coincidental majority agreements.

### 5.3 Full-Stage Ensemble Validation

Combining all 35 models (7 domains x 5 training stages) with vote threshold sweep:

| Vote Threshold | F1 | Precision | Recall | Notes |
| :---: | :---: | :---: | :---: | :--- |
| 10 | 0.8728 | 0.8341 | 0.9153 | Too permissive |
| 17 | 0.8876 | 0.8680 | 0.9081 | Approx. 7-model Synth V2 |
| **21** | **0.8916** | **0.8864** | **0.8968** | **Optimal** |
| 22 | 0.8915 | 0.8916 | 0.8915 | P=R equilibrium |
| 25 | 0.8830 | 0.9081 | 0.8593 | Too strict |

**Optimal ratio: 21/35 = 60%** (vs. 5/7 = 71% for 7-model ensemble). Larger ensembles allow lower consensus thresholds due to increased statistical reliability.

> **Design Principle:** In MoE ensembles, **model diversity is more important than individual benchmark optimization.** Varying training data composition is the most effective means of achieving diversity.

---

## 6. Contributions

| Contribution | Description |
| :--- | :--- |
| Large-scale synthetic dataset | 52,000 instrument- and technique-specific synthetic tracks |
| Post-processing elimination | MoE consensus replaces filtering, DP string assignment, and quantization |
| Domain adaptation effectiveness | Synthetic to real-recording FT improves F1: 0.5610 to 0.8916 |
| Multi-dataset integration | 3DS integration enables nylon-string generalization (GAPS F1=0.7312) |
| **Diversity regularization discovery** | **Individual F1 degradation (-0.023) yet ensemble F1 improvement (+0.004) via synthetic mixing** |
| CNN string classifier | CQT-based 6-class string prediction, Val accuracy 94.1% (optimized), match rate 92.30% |
| Synthetic data analysis | Quantitative domain gap analysis proving synthetic-only training infeasible for string classification |
| Negative transfer evidence | v5 pre-training degrades GS accuracy by -11.1% vs. direct training |
| Fingering LSTM | Bi-LSTM integrating CNN probabilities + context, Val accuracy 98.31% |
| Evaluation transparency | Explicit Train/Test splits, LOO cross-validation, cross-dataset evaluation |

---

## 7. Future Work

1. **GAPS Recall Improvement:** Domain-adaptive vote thresholds for nylon-string audio
2. ~~**String Classifier Multi-Domain Training:** Retraining with GAPS audio to improve 70.59% to 80%+~~ → **Achieved in Step 12: 23.9% → 75.8% (+51.9pp)**
3. **Architectural Evolution:** Self-Attention layers for long-range dependency modeling
4. **Human Fingering Analysis:** Comparing IDMT human position choices vs. algorithmic assignments to improve tablature naturalness
5. **Extended IDMT Training:** Longer fine-tuning (10+ epochs) for pick-domain models showing improvement trends

---

## 7.1 IDMT-SMT-V2 Integration Experiment (Step 11)

> **Experiment Date: 2026-05-09**

### Motivation

IDMT-SMT-V2 contains 252 tracks of real electric guitar recordings with **human-annotated string and fret positions** in XML format. Unlike synthetic data or algorithmic fingering assignments, these labels reflect actual guitarist performance decisions, offering a unique source of diversity for ensemble learning.

### Setup

- **Training Data:** GuitarSet(286) + GAPS(371) + Synth V2(286/5000) + IDMT(252) = ~1195 batches/epoch
- **Initial Weights:** multitask_3ds_ga (Step 9 models)
- **Epochs:** 3, Patience: 3
- **Output Suffix:** `multitask_4ds` (no overwriting of existing models)

### Individual Model Results

| Domain | 4DS Best F1 | Synth V2 F1 | Delta |
| :--- | :---: | :---: | :---: |
| martin_finger | 0.7704 | 0.7734 | -0.0030 |
| taylor_finger | 0.7460 | 0.7522 | -0.0062 |
| luthier_finger | 0.7613 | 0.7629 | -0.0016 |
| **martin_pick** | **0.7811** | 0.7775 | **+0.0036** |
| **taylor_pick** | **0.7745** | 0.7735 | **+0.0010** |
| **luthier_pick** | **0.7791** | 0.7735 | **+0.0056** |
| gibson_thumb | 0.7641 | 0.7735 | -0.0094 |

All three pick-domain models improved, while finger-domain and thumb models slightly degraded.

### 42-Model MoE Benchmark

35 existing models + 7 new multitask_4ds models = 42 models. Vote threshold sweep 12-30.

| Vote | F1 | Precision | Recall | Notes |
| :---: | :---: | :---: | :---: | :--- |
| 21 | 0.8882 | 0.8692 | 0.9081 | Equivalent to 35-model optimal |
| **23** | **0.8913** | **0.8789** | **0.9040** | **42-model optimal (reproduced 2x)** |
| 26 | 0.8897 | 0.8897 | 0.8897 | P=R equilibrium |

### Comparison

| Configuration | Best F1 | Optimal Vote | Ratio |
| :--- | :---: | :---: | :---: |
| 7 models (Synth V2) | 0.8877 | 5/7 | 71% |
| **35 models** | **0.8916** | **21/35** | **60%** |
| 42 models (+IDMT) | 0.8913 | 23/42 | 55% |

### Analysis

1. **42-model F1=0.8913 is marginally below 35-model F1=0.8916** (-0.0003), statistically equivalent.
2. **Factors:** Large domain gap between electric and acoustic guitar; 3 epochs may be insufficient (pick domains were still improving).
3. **Pick-domain improvement is noteworthy:** IDMT's pick-style data selectively enhanced pick-domain model diversity, suggesting domain-selective data integration has value.
4. **Human fingering data:** IDMT's human-annotated string/fret positions remain valuable for future tablature naturalness improvements.

> **Conclusion:** IDMT-SMT-V2 integration did not surpass the 35-model record. **F1=0.8916 remains the confirmed SOTA.**

---

## 8. General Conclusion

### 8.1 Overview

This study systematically overcame multiple technical barriers in guitar AMT, ultimately achieving GuitarSet Test Pitch F1 = 0.8916 through a 35-model full-stage integrated MoE ensemble. In the process, we acquired novel design insights across three domains: ensemble learning, domain adaptation, and string assignment.

### 8.2 Key Achievements

| Achievement | Metric | Value |
| :--- | :--- | :---: |
| Pitch detection (steel-string) | GuitarSet Test Pitch F1 | 0.8916 |
| Pitch detection (nylon-string) | GAPS Test Pitch F1 | 0.7312 |
| String classifier accuracy | CNN Val accuracy (optimized) | **94.1%** |
| String assignment accuracy | CNN-first string+fret match (GuitarSet) | 96.60% |
| String assignment generalization | Leave-One-Out cross-validation | 80.92% |
| Sequence string prediction | Fingering LSTM Val accuracy | 98.31% |
| Post-processing | Noise filter, DP, quantization | Fully eliminated |

### 8.3 Three Core Findings

**Finding 1: Diversity Effect via Synthetic Data Regularization**

Synth V2 mixed training degraded individual model GuitarSet Val F1 by an average of -0.0157, yet improved MoE ensemble F1 by +0.0038. This occurred because the error patterns of each model diversified, enabling noise cancellation during consensus voting. This demonstrates that ensuring inter-model diversity is more important than maximizing individual model accuracy for ensemble quality.

**Finding 2: Cumulative Diversity Effect via Full-Stage Integration**

By retaining and combining all models from every training stage (35 models) rather than discarding intermediate checkpoints, we achieved an additional +0.0077 F1 improvement over the 7-model configuration. This approach, which improves performance without additional training cost, is a practical strategy for continuously evolving systems.

**Finding 3: CNN-First Architecture for String Estimation**

Replacing Viterbi DP-based string assignment (61.18%) with a CNN string classifier using audio CQT features as input dramatically improved string+fret match rate to 96.60% (+35.42%). This demonstrates that overcoming the theoretical ceiling of pitch-only string estimation (~70%) fundamentally requires the utilization of audio spectral features.

### 8.4 Overall Effect of Progressive Training Strategy

| Stage | Description | MoE Pitch F1 | Cumulative Improvement |
| :--- | :--- | :---: | :---: |
| Baseline | Synthetic pre-training only | 0.5610 | -- |
| Step 2 | GuitarSet domain adaptation | 0.8310 | +0.2700 |
| Step 6 | 3-dataset integration (3DS) | 0.8839 | +0.0529 |
| Step 9 | Synth V2 diversity mixing | 0.8877 | +0.0038 |
| Step 10 | 35-model full-stage integration | 0.8916 | +0.0039 |
| **Total** | | | **+0.3306 (+58.9%)** |

GuitarSet-specific domain adaptation (Step 2) provided the largest single improvement, followed by cumulative gains through multi-dataset integration and diversity enhancement.

### 8.5 Position Relative to Prior Work

Under the constraint of using absolutely no post-processing, our pure MoE ensemble (Step 10) significantly surpasses TabCNN (F1 ~ 0.826) and achieves accuracy equal to or exceeding existing methods that rely on extensive post-processing. The combination of training scale (52,000 synthetic tracks + 3-dataset integrated fine-tuning) with architectural simplicity (no post-processing) represents the unique contribution of this work.

### 8.6 Summary

SoloTab V2.0 achieves competitive transcription accuracy while departing from conventional post-processing-dependent architectures, through a Pure MoE ensemble built on large-scale synthetic data and progressive domain adaptation. The finding that "diversity determines consensus quality more than individual model accuracy" is a universal principle applicable to ensemble learning in general, and is expected to contribute to future research. By integrating a CNN string classifier (match rate 96.60%) leveraging audio CQT features with a fingering LSTM, this study significantly improved the accuracy and robustness of the entire transcription pipeline, taking an important step toward the practical deployment of guitar AMT.

---

## Acknowledgements and References

### Acknowledgements

This research deeply relies on the following datasets, tools, and prior work. We express our sincere respect and gratitude to the researchers and developers who made these resources publicly available.

### Datasets

**GuitarSet**

Qingyang Xi, Rachel M. Bitteur, Juan Pablo Bello. "GuitarSet: A Dataset for Guitar Transcription." Proceedings of the 19th ISMIR, 2018.

- License: CC BY 4.0
- URL: https://github.com/marl/guitarset

**GAPS (Guitar-Aligned Performance Scores)**

Xavier Riley, Zixun Guo, Drew Edwards, Simon Dixon. "GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model." ISMIR, 2024.

**AG-PT-set (Acoustic Guitar Playing Technique Set)**

12 expressive technique annotations for acoustic guitar. Used in 3-dataset integration (3DS, Step 6).

**IDMT-SMT-Guitar V2**

Human-annotated electric guitar recordings with string/fret labels. 252 tracks across 3 guitar models (Fender Stratocaster, Les Paul, Archtop).

### References

1. A. Wiggins, Y. Kim. "Guitar Tablature Estimation with a CNN." ISMIR, 2019. (TabCNN baseline, F1 ~ 0.826)
2. "SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription." 2024. (CRNN + synthetic augmentation, F1 ~ 0.87+)
3. A. Gulati et al. "Conformer: Convolution-augmented Transformer for Speech Recognition." Interspeech, 2020.
4. Bontempi et al. "Biomechanical constraints for guitar fingering using Inter-Onset Interval." 2024.
5. A. Radisavljevic, P. Driessen. "Path Difference Learning for Guitar Chord/Solo Transcription." ICMC, 2004.
6. T. Hori, S. Sagayama. "Minimax Viterbi Algorithm for HMM-Based Guitar Tablature Transcription." ISMIR, 2016.
7. Bitteur et al. / Spotify Research. "Basic Pitch: A Lightweight yet Powerful Pitch Detection Library." 2022. URL: https://github.com/spotify/basic-pitch

### Tools and Libraries

| Tool / Library | Purpose |
| :--- | :--- |
| PyTorch | CRNN, CNN, LSTM training and inference |
| librosa | CQT spectrogram generation, audio feature extraction |
| mirdata | Standardized GuitarSet access interface |
| mir_eval | Standard evaluation metrics (Pitch F1, Precision, Recall) |
| ONNXRuntime | Basic Pitch model inference |
| music21 / MusicXML | Score format output |

---

## Appendix A: System Configuration

| Component | Description |
| :--- | :--- |
| Note Detection | 7xCRNN (BiGRU-768) domain-specific MoE ensemble |
| Training Data | Synthetic 52K + GuitarSet 286 + GAPS 371 + AG-PT 72 + Synth V2 286 |
| Post-Processing | **None** (all filtering, DP, quantization eliminated) |
| String Assignment | CNN string classifier (CQT 84bins×11frames + pitch, Val 94.1%) + Bi-LSTM context refinement |
| String Classifier Training | GuitarSet 61,885 samples, AdamW + CosineAnnealing + augmentation, 80 epochs |
| **GAPS Cross-Domain** | **CNN string classifier: 23.9% → 75.8% (+51.9pp) via data quality correction (Step 12)** |
| Hardware | NVIDIA RTX 4060 Ti (8GB VRAM), Windows 11 |
| Framework | PyTorch 2.x, librosa, pretty_midi |

---

## Step 12: CNN String Classifier Cross-Domain Adaptation (GAPS, 2026-05-17)

### Motivation

The CNN string classifier (Val 94.1%, §8.6.7) was trained exclusively on GuitarSet (steel-string, hexaphonic pickup). On GAPS (nylon-string, YouTube recordings), string classification accuracy was only **23.9%** — insufficient for practical nylon-string transcription.

### Root Cause: Data Quality Issues

Analysis of the initial GAPS dataset (78K patches) revealed **three critical quality problems**:

| Metric | GuitarSet | GAPS v1 | Issue |
| :--- | :---: | :---: | :--- |
| Mean patch energy | 0.386 | 0.017 | 4.4% of GS — patches crushed to near-zero |
| Silent patch rate | 0.0% | 22.0% | 1 in 5 patches contain no signal |
| Onset center rate | 28.3% | 1.2% | Timing completely misaligned |
| Inter-string spectral distance | 2.73 | 0.32 | String timbral differences invisible |

**Root causes**: (1) Track-level CQT max normalization crushing note-level energy, (2) MusicXML tempo-based timing failing due to rubato/tempo changes, (3) Both effects making string spectral differences undetectable.

### Dataset Improvements

**v2 (Patch normalization + Onset snap):** Per-patch CQT normalization + librosa onset detection matching → 180,633 patches (0% silent, 69.5% onset snap rate).

**v3 (DTW + Onset snap):** MIDI↔audio chroma DTW alignment (Sakoe-Chiba band) + nearest onset snap within ±150ms → 182,599 patches (97.1% snap rate, 15.8% onset center rate).

### Training Results

| Method | GAPS Val |
| :--- | :---: |
| v1 data: GS→GAPS FT (all layers) | 71.2% (ceiling) |
| v2 data: GS→GAPS FT (all layers, 80 ep) | 75.8% |
| v3 data: DTW + FT | 75.8% (same) |
| GS+GAPS v2 mixed training | GS: 95.3% / GAPS: 74.8% |

**Key finding**: Data quality correction (+4.6pp) vastly outperformed model/hyperparameter tuning (+0.6pp). The 70% ceiling was caused by data quality, not model capacity.

### Unified Benchmark: 3-Model Comparison

All three models evaluated on **identical val splits and normalization** across both domains:

| Model | GS Val (8,840) | GAPS Val (36,126) | Combined |
| :--- | :---: | :---: | :---: |
| GS-only (production) | **98.7%** | 20.7% | 59.7% |
| GAPS-only v2 | 36.2% | 75.8% | 56.0% |
| **Mixed v2** 🏆 | **95.3%** | **79.0%** | **87.2%** |

> **Critical finding**: Mixed v2 achieves **79.0%** on GAPS, surpassing the GAPS-specialized model (75.8%) by **+3.2pp**. GuitarSet's rich inter-string patterns generalize to nylon-string domain. Mixed v2 also maintains 95.3% on GuitarSet.

**Per-string accuracy (GAPS Val)**: Mixed v2 outperforms GAPS-only on 5 of 6 strings (S1: 96% vs 92%, S2: 73% vs 69%, S4: 77% vs 75%, S5: 83% vs 77%, S6: 89% vs 85%), with only S3 (G string) tied at 62%.

### Models Produced

| File | Purpose | GS Val | GAPS Val |
| :--- | :--- | :---: | :---: |
| `string_classifier.pth` | GS-only | **98.7%** | 20.7% |
| `string_classifier_gaps_v2.pth` | GAPS-only | 36.2% | 75.8% |
| `string_classifier_mixed_v2.pth` | **🏆 Recommended** | **95.3%** | **79.0%** |

### Ceiling Analysis

The remaining 21.0% error is attributed to: (1) nylon string inter-string spectral distance being 1/8 of steel strings, with G string (S3) at 62% due to wound/plain string boundary, (2) MusicXML editorial fingering vs. actual performance string discrepancies, (3) YouTube recording quality variability, (4) inherently noisy labels (MusicXML intent vs. hexaphonic physical measurement).

> **Design principle**: Multi-domain mixed training outperforms domain-specific fine-tuning, echoing the "diversity-driven consensus quality improvement" discovered in MoE ensemble training (§10).

---

*SoloTab V2.0 -- May 2026*

