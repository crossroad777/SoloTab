import sys
import torch
import warnings
warnings.filterwarnings("ignore")

import ensemble_transcriber
import os

def test():
    wav_path = r"d:\Music\nextchord-solotab\禁じられた遊び　(ロマンス ) ギター Tab譜 楽譜　コードネーム付 - アコースティック 名曲 ギター タブ 楽譜ギター タブ譜 (128k).wav"
    
    if not os.path.exists(wav_path):
        print("Test wav not found")
        return
        
    print(f"Testing on {os.path.basename(wav_path)}")
    res = ensemble_transcriber.transcribe_ensemble(wav_path)
    print("Done!")
    
    # MIDI 64 (E4) の検出状況を確認
    e4_notes = [n for n in res['notes'] if n['pitch'] == 64]
    print(f"Found {len(e4_notes)} E4 notes.")
    if e4_notes:
        print("First 5 E4 notes:")
        for n in e4_notes[:5]:
            print(n)

if __name__ == "__main__":
    test()
