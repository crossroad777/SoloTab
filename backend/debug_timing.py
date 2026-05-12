import json

with open(r"D:\Music\nextchord-solotab\uploads\20260512-073742\beats.json") as f:
    bd = json.load(f)

beats = bd["beats"]
bpm = bd["bpm"]
ts = bd.get("time_signature", "?")

print(f"BPM: {bpm}, Time sig: {ts}, Beats: {len(beats)}")
print(f"First beat: {beats[0]:.3f}s, Last: {beats[-1]:.3f}s")
print(f"Total duration: {beats[-1] - beats[0]:.1f}s")
print(f"Expected interval at {bpm} BPM: {60/bpm:.3f}s")
print(f"Actual avg interval: {(beats[-1]-beats[0])/(len(beats)-1):.3f}s")
print()

# Check if audio file duration matches beat coverage
import wave
wav_path = r"D:\Music\nextchord-solotab\uploads\20260512-073742\converted.wav"
try:
    with wave.open(wav_path, 'r') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        audio_dur = frames / rate
        print(f"Audio duration: {audio_dur:.1f}s")
        print(f"Beats cover: {beats[0]:.1f}s to {beats[-1]:.1f}s")
        print(f"Beat coverage: {(beats[-1]-beats[0])/audio_dur*100:.1f}% of audio")
except:
    print("Could not read audio")

print()
print("First 12 beats (Bars 0-3 of 3/4):")
for i in range(min(12, len(beats))):
    interval = beats[i+1]-beats[i] if i+1 < len(beats) else 0
    bar = i // 3
    beat_in_bar = (i % 3) + 1
    print(f"  beat[{i:3d}] = {beats[i]:7.3f}s  (Bar {bar}, beat {beat_in_bar}, dt={interval:.3f}s)")

# Calculate what the BeatMap would produce
print()
print("=== BeatMap timing check ===")
print("bars = 71, beats_per_bar = 3")
for bar_num in range(4):
    beat_idx = bar_num * 3
    if beat_idx >= len(beats):
        break
    start_ms = beats[beat_idx] * 1000
    end_idx = beat_idx + 3
    if end_idx < len(beats):
        end_ms = beats[end_idx] * 1000
    else:
        end_ms = start_ms + 3 * 600
    print(f"  Bar {bar_num}: {start_ms:.0f}ms - {end_ms:.0f}ms (dur={end_ms-start_ms:.0f}ms)")
