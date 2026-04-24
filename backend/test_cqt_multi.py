import os
import glob
import torch

# sys.path への追加等が必要な場合は precompute_cqt_caches 内部で解決されているはずですが、
# importが通るようにbackendディレクトリ内で実行します。
from precompute_cqt_caches import compute_cqt

def test_single_file():
    dataset_dir = r"D:\Music\datasets"
    wav_files = list(glob.iglob(os.path.join(dataset_dir, "GuitarSet", "**", "*_mic.wav"), recursive=True))
    if not wav_files:
        print("No GuitarSet files found.")
        return

    test_file = wav_files[0]
    print(f"Testing on {test_file}")
    
    cache_path = test_file + ".cqt_multi.pt"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        
    compute_cqt(test_file)
    
    if os.path.exists(cache_path):
        tensor = torch.load(cache_path, weights_only=True)
        print(f"Success! Tensor shape: {tensor.shape}")
        if tensor.shape[0] == 3 and tensor.shape[1] == 168:
            print("Shape is correct: [3, 168, T]")
        else:
            print("Shape is INCORRECT!")
    else:
        print("Failed to generate cache file.")

if __name__ == "__main__":
    test_single_file()
