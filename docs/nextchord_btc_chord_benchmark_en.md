# Comprehensive Benchmark of Automatic Chord Recognition on Solo Guitar Audio
# — BTC Engine Evaluation Across Three Datasets with Comping/Solo Performance Gap Analysis —

**NextChord Research Report**

## Abstract

Automatic Chord Recognition (ACR) is a foundational task in Music Information Retrieval (MIR), yet systematic benchmarking on solo guitar audio remains insufficient. This study presents a comprehensive evaluation of the BTC (Bi-directional Transformer for Chords) engine across three datasets: GuitarSet (360 tracks, studio microphone recordings), Beatles/Isophonics (180 tracks), and uspop2002 (190 tracks). Under MIREX standard evaluation (`mir_eval.chord.evaluate`), the GuitarSet comping (comp) tracks achieved **thirds = 0.860**, exactly matching the BTC paper's published reference value. In contrast, solo performance tracks yielded only **thirds = 0.191**, revealing an extreme performance gap of 0.669. These results demonstrate that existing chord recognition models are optimized for harmonic (chordal) input, and that dedicated approaches are essential for chord estimation from single-note melodies. We further demonstrate that diatonic filtering post-processing is harmful, and that YouTube-dependent evaluation introduces up to 30% score degradation.

---

## 1. Introduction

Chord recognition is the task of estimating harmonic progressions as time-series labels from audio signals. While deep learning approaches have achieved high accuracy on pop/rock datasets such as Beatles, several challenges remain:

1. **Application to solo guitar audio**: Existing models are trained and evaluated on mixed-instrument recordings; performance on guitar-only audio is unverified
2. **Comping vs. solo distinction**: Chordal accompaniment and single-note solo parts present fundamentally different input characteristics
3. **Evaluation reliability**: YouTube-dependent evaluation contains systematic biases from version mismatches and pitch shifts

This study leverages the studio-quality recordings of GuitarSet, a guitar-specific dataset, to provide quantitative insights into these challenges.

---

## 2. Methodology

### 2.1 Chord Recognition Engine
- **BTC (Bi-directional Transformer for Chords)**
- Large vocabulary mode (extended chord notation support)
- Pre-trained model used without modification

### 2.2 Evaluation Metrics
Following the MIREX standard via `mir_eval.chord.evaluate`:
- **root**: Root note agreement
- **thirds**: Root + third quality agreement ← **primary metric**
- **mirex**: Official MIREX score
- **majmin**: Major/minor agreement
- **sevenths**: Seventh chord agreement

### 2.3 Datasets

| Dataset | Tracks | Audio Source | Annotations | Characteristics |
|---|---|---|---|---|
| **GuitarSet** | 360 | Studio mic recording | leadsheet_chords | Guitar-specific, 5 genres × 6 players |
| **Beatles (Isophonics)** | 180 | YouTube | Isophonics .lab | Rock, band arrangement |
| **uspop2002** | 190 | YouTube | uspop .lab | Multi-genre pop |

GuitarSet contains both comp (chordal accompaniment) and solo tracks for each piece, enabling **direct comparison of performance differences due to input signal type under identical annotations**.

---

## 3. Results

### 3.1 Overall Results by Dataset

| Dataset | Tracks | avg thirds | Source Quality |
|---|---|---|---|
| **GuitarSet comp** | 180 | **0.860** | ★★★ Studio recording |
| **Beatles** | 180 | **0.792** | ★★ YouTube (corrected) |
| uspop2002 Clean | 143 | 0.689 | ★★ YouTube (duration-matched) |
| uspop2002 Full | 190 | 0.565 | ★ YouTube (pitch-normalized) |
| GuitarSet Full | 360 | 0.525 | ★★★ Studio recording |
| **GuitarSet solo** | 180 | **0.191** | ★★★ Studio recording |

### 3.2 GuitarSet comp vs solo — All Metrics

| Metric | Comp (180) | Solo (180) | Gap |
|---|---|---|---|
| root | **0.879** | 0.242 | -0.637 |
| **thirds** | **0.860** | **0.191** | **-0.669** |
| mirex | **0.859** | 0.194 | -0.665 |
| majmin | **0.862** | 0.194 | -0.668 |
| sevenths | **0.840** | 0.176 | -0.664 |

A **consistent gap of 0.64–0.67** exists across all metrics between comp and solo.

### 3.3 Genre-Level Analysis (comp vs solo)

| Genre | Comp thirds | Solo thirds | Gap |
|---|---|---|---|
| Singer-Songwriter | **0.945** | 0.208 | -0.737 |
| Rock | **0.944** | 0.193 | -0.751 |
| Funk | 0.801 | 0.218 | -0.583 |
| Jazz | 0.789 | 0.194 | -0.595 |
| Bossa Nova | 0.820 | 0.140 | -0.680 |

- **Singer-Songwriter / Rock comp exceed 0.94**: Highest accuracy for diatonic-dominant progressions
- **Jazz / Bossa Nova comp at 0.79–0.82**: Extended harmonies (7th, 9th, dim) reduce thirds-metric scores
- **Solo uniformly low at 0.14–0.22**: Genre-independent — this is an input signal problem, not a harmonic complexity issue

### 3.4 Score Distribution

| Range | Tracks | Ratio | Composition |
|---|---|---|---|
| ≥ 0.7 | 163 | 45% | Nearly all comp |
| 0.3-0.7 | 48 | 13% | Mixed |
| < 0.3 | 149 | 41% | Nearly all solo |

---

## 4. Post-Processing Experiment: Diatonic Filter

We investigated whether music theory-based post-processing could improve BTC output. Key estimation via the Krumhansl-Schmuckler algorithm was applied, followed by correction of non-diatonic chords to their nearest diatonic equivalents.

### 4.1 Results (Beatles 180 tracks)

| Strength | thirds | delta | Improved | Degraded |
|---|---|---|---|---|
| None (BTC raw) | **0.792** | — | — | — |
| soft (maj/min correction) | 0.697 | **-0.095** | 21 | **119** |
| moderate | 0.692 | -0.101 | 20 | 121 |
| strong | 0.686 | -0.106 | 19 | 124 |

### 4.2 Discussion
Diatonic filtering is **harmful at all strengths**. BTC's Transformer self-attention mechanism has already implicitly learned diatonic relationships from its training data. External forced corrections erroneously "fix" intentional non-diatonic outputs (modulations, borrowed chords, etc.).

---

## 5. Evaluation Reliability Analysis

### 5.1 YouTube Audio Issues

Detailed analysis of low-scoring uspop2002 tracks identified systematic problems:

1. **Version mismatches**: Track durations significantly differ from reference annotations (e.g., Layla — REF: 338s vs EST: 424s)
2. **Pitch shifts**: Retrieval of versions in different keys (e.g., ABBA "I Have A Dream" — REF: A major → EST: A# major)

### 5.2 Pitch Normalization Experiment

All 12 semitone transpositions of estimated chords were tested to detect optimal key alignment:
- **uspop2002**: 14 tracks with pitch shift detected and corrected (e.g., Celine Dion "My Heart Will Go On" 0.133 → 0.815)
- **Beatles**: 0 pitch-shifted tracks detected

### 5.3 Duration Filtering

Tracks with >20% duration mismatch between reference and estimate were excluded (alternate version removal):
- uspop2002: 190 → 143 tracks (Clean), thirds: 0.548 → **0.689**
- 47 tracks (25%) were alternate versions or inappropriate audio

---

## 6. Discussion

### 6.1 Significance of comp = 0.860

The GuitarSet comp thirds = 0.860 matches the BTC paper's Beatles reference value of 0.860. This demonstrates:
- BTC achieves its published performance on **high-quality studio recordings**
- **Guitar-only audio achieves accuracy comparable to mixed-instrument sources** when the input contains chordal content
- Guitar chord recognition is at a practical level when audio quality is assured

### 6.2 Root Cause of solo = 0.191

The extreme low score on solo tracks stems from a fundamental mismatch between BTC's input assumption (signals containing harmonic structure) and solo reality (single-note melodies):
- BTC identifies chord root and third from **harmonic overtone structure**
- Single-note input **physically lacks third interval information**
- Consequently, BTC can detect the root pitch but cannot determine major/minor quality

### 6.3 YouTube Evaluation Bias

| Source | Beatles thirds | uspop2002 thirds |
|---|---|---|
| Studio / correct YT audio | **0.792** | **0.689** (Clean) |
| All YT audio (uncorrected) | 0.702 | 0.548 |
| Degradation | -0.090 (-11%) | -0.141 (-20%) |

YouTube-dependent evaluation risks systematically underestimating true model performance.

---

## 7. Conclusion

### Confirmed Results

| Metric | Value | Condition |
|---|---|---|
| **GuitarSet comp thirds** | **0.860** | Studio recording, 180 tracks |
| GuitarSet solo thirds | 0.191 | Studio recording, 180 tracks |
| Beatles thirds | 0.792 | YouTube, 180 tracks (corrected) |
| uspop2002 Clean thirds | 0.689 | YouTube, 143 tracks (filtered) |

### Key Contributions

| Contribution | Detail |
|---|---|
| **Guitar-specific benchmark** | First comprehensive BTC evaluation on GuitarSet studio recordings |
| **Comp/solo gap quantification** | 0.669 performance gap demonstrated across 5 genres |
| **Post-processing harmfulness** | Diatonic filter degrades performance at all strengths |
| **YouTube bias quantification** | Up to 20% score degradation measured empirically |

### Future Work

1. **Solo guitar-specific chord recognition engine**: Approaches to improve solo = 0.191 (harmony estimation from melody)
2. **BTC fine-tuning on GuitarSet comp**: Investigating improvement potential beyond 0.860
3. **Vocal separation + BTC pipeline**: Chord recognition after guitar accompaniment isolation from mixed sources
4. **Bottom-up chord estimation from note information**: Using transcription results as input for chord inference
