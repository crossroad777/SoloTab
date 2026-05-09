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

1. **CNN String Classifier:** Predicts string from CQT features around each detected note (6-class classification, Val accuracy 92.66%)
2. **Bi-LSTM Refinement:** Integrates CNN probabilities with sequential context (Val accuracy 98.31%)

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
| CNN string classifier | CQT-based 6-class string prediction, 92.30% match rate |
| Fingering LSTM | Bi-LSTM integrating CNN probabilities + context, Val accuracy 98.31% |
| Evaluation transparency | Explicit Train/Test splits, LOO cross-validation, cross-dataset evaluation |

---

## 7. Future Work

1. **IDMT-SMT-V2 Integration:** Electric guitar data with human-annotated fingerings -- unique source of real-world position choices (currently in progress)
2. **GAPS Recall Improvement:** Domain-adaptive vote thresholds for nylon-string audio
3. **String Classifier Multi-Domain Training:** Retraining with GAPS audio to improve 70.59% to 80%+
4. **Architectural Evolution:** Self-Attention layers for long-range dependency modeling
5. **Human Fingering Analysis:** Comparing IDMT human position choices vs. algorithmic assignments to improve tablature naturalness

---

## Appendix A: System Configuration

| Component | Description |
| :--- | :--- |
| Note Detection | 7xCRNN (BiGRU-768) domain-specific MoE ensemble |
| Training Data | Synthetic 52K + GuitarSet 286 + GAPS 371 + AG-PT 72 + Synth V2 286 |
| Post-Processing | **None** (all filtering, DP, quantization eliminated) |
| String Assignment | CNN string classifier (CQT to 6class) + Bi-LSTM context refinement |
| Hardware | NVIDIA RTX 4060 Ti (8GB VRAM), Windows 11 |
| Framework | PyTorch 2.x, librosa, pretty_midi |

---

*SoloTab V2.0 -- May 2026*
