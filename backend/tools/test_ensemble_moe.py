import sys
import torch
import warnings
warnings.filterwarnings("ignore")

import ensemble_transcriber
import os
import glob

def test():
    wav_path = r"D:\Music\Datasets\GuitarSet\audio_mono-pickup_mix\00_BN1-129-Eb_comp_mix.wav"
    files = glob.glob(r"D:\Music\Datasets\GuitarSet\audio_mono-pickup_mix\*.wav")
    if files:
        wav_path = files[0]
    else:
        print("Test wav not found in GuitarSet")
        # 他の場所を探す
        return
        
    print(f"Testing on {wav_path}")
    res = ensemble_transcriber.transcribe_ensemble(wav_path)
    print("Done!")
    print(res["model_stats"])
    
if __name__ == "__main__":
    test()
