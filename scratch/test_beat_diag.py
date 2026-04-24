import sys
import os
from pathlib import Path

# パス追加
sys.path.append(r"D:\Music\nextchord-solotab\backend")

from beat_detector import detect_beats

wav_path = r"D:\Music\nextchord-solotab\uploads\20260413-212229-yt-abca7e\converted.wav"

print(f"Testing beat detection on: {wav_path}")
if not os.path.exists(wav_path):
    print("Error: File not found")
    sys.exit(1)

try:
    import time
    start = time.time()
    result = detect_beats(wav_path)
    end = time.time()
    print(f"Success! BPM: {result['bpm']}, Beats: {len(result['beats'])}")
    print(f"Time taken: {end - start:.2f}s")
except Exception as e:
    print(f"Failed with error: {e}")
    import traceback
    traceback.print_exc()
