---
title: SoloTab - Guitar Tablature Transcriber
emoji: 🎸
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# SoloTab V2.0 — Pure MoE Guitar Tablature Transcriber

Upload a solo guitar audio file and get an accurate tablature transcription.

## How it works

1. **CQT Feature Extraction** — Constant-Q Transform spectrogram is computed from the audio.
2. **6-Expert MoE Inference** — Six domain-specific CRNN models (Martin/Taylor/Luthier × Finger/Pick) analyze the audio independently.
3. **Majority Voting** — Notes are detected only when 5+ experts agree, ensuring high precision.
4. **Tablature Generation** — Detected notes are rendered as guitar tablature with string and fret information.

## Performance

- **F1 = 0.8310** on GuitarSet test split (36 tracks)
- **F1 = 0.8478** on full GuitarSet (360 tracks, reference)

## Paper

See our research paper for full details on the Pure MoE architecture and training methodology.
