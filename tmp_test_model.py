import sys
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")
from guitar_transcriber import _get_model_paths, _load_model

try:
    print("Paths:", _get_model_paths())
    model, device = _load_model()
    print("Model loaded OK on", device)
    
except Exception as e:
    print("Error:", e)
