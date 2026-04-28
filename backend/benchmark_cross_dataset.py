# -*- coding: utf-8 -*-
"""Cross-dataset benchmark: GuitarSet-trained CNN on IDMT-SMT-Guitar"""
import warnings
warnings.filterwarnings('ignore')
import sys, os, glob, torch, numpy as np, librosa
import xml.etree.ElementTree as ET

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
from string_classifier import StringClassifierCNN, N_BINS, CONTEXT_FRAMES, SR, HOP_LENGTH

IDMT_BASE = r'D:\Music\datasets\IDMT-SMT-GUITAR\IDMT-SMT-GUITAR_V2'
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

# IDMT stringNumber: 1=6th(low E), 6=1st(high E)
# GuitarSet/SoloTab: 1=1st(high E), 6=6th(low E)
def idmt_to_solotab_string(idmt_string):
    return 7 - idmt_string

def load_model():
    device = torch.device('cpu')
    model = StringClassifierCNN().to(device)
    state = torch.load(r'D:\Music\nextchord-solotab\backend\string_classifier.pth',
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    notes = []
    for event in root.findall('.//event'):
        pitch = int(event.find('pitch').text)
        onset = float(event.find('onsetSec').text)
        offset = float(event.find('offsetSec').text)
        fret = int(event.find('fretNumber').text)
        string_idmt = int(event.find('stringNumber').text)
        string_st = idmt_to_solotab_string(string_idmt)
        notes.append({
            'pitch': pitch, 'onset': onset, 'offset': offset,
            'fret': fret, 'string': string_st, 'string_idmt': string_idmt
        })
    return notes

def predict_string(model, device, audio, sr_audio, onset_sec, midi_pitch):
    if sr_audio != SR:
        audio = librosa.resample(audio, orig_sr=sr_audio, target_sr=SR)
        sr_audio = SR
    
    cqt_spec = librosa.cqt(y=audio, sr=SR, hop_length=HOP_LENGTH,
                            fmin=librosa.note_to_hz('C2'), n_bins=N_BINS, bins_per_octave=12)
    cqt_mag = np.abs(cqt_spec)
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    
    onset_frame = int(onset_sec * SR / HOP_LENGTH)
    half_ctx = CONTEXT_FRAMES // 2
    
    if onset_frame - half_ctx < 0:
        onset_frame = half_ctx
    if onset_frame + half_ctx >= cqt_db.shape[1]:
        onset_frame = cqt_db.shape[1] - half_ctx - 1
    if onset_frame - half_ctx < 0 or onset_frame + half_ctx >= cqt_db.shape[1]:
        return -1
    
    patch = cqt_db[:, onset_frame - half_ctx:onset_frame + half_ctx + 1]
    patch_t = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
    pitch_t = torch.FloatTensor([[( midi_pitch - 40) / 45.0]]).to(device)
    
    with torch.no_grad():
        out = model(patch_t, pitch_t)
        pred = out.argmax(dim=1).item() + 1
    return pred

def main():
    model, device = load_model()
    print("Model loaded (CPU)")
    
    # Collect all XML+WAV pairs
    all_correct = 0
    all_total = 0
    by_dataset = {}
    
    for ds_name in ['dataset1', 'dataset2', 'dataset3']:
        ds_path = os.path.join(IDMT_BASE, ds_name)
        if not os.path.exists(ds_path):
            continue
        
        xml_files = glob.glob(os.path.join(ds_path, '**', '*.xml'), recursive=True)
        print("\n--- %s: %d XML files ---" % (ds_name, len(xml_files)))
        
        ds_correct = 0
        ds_total = 0
        
        for xml_path in xml_files[:50]:  # Limit to 50 per subset for speed
            wav_path = xml_path.replace('.xml', '.wav')
            if not os.path.exists(wav_path):
                # annotation/ -> audio/ pattern
                wav_name = os.path.basename(xml_path).replace('.xml', '.wav')
                parent = os.path.dirname(xml_path)
                if os.path.basename(parent) == 'annotation':
                    wav_path = os.path.join(os.path.dirname(parent), 'audio', wav_name)
                if not os.path.exists(wav_path):
                    continue
            
            notes = parse_xml(xml_path)
            if not notes:
                continue
            
            audio, sr_audio = librosa.load(wav_path, sr=None, mono=True)
            
            for note in notes:
                pred = predict_string(model, device, audio, sr_audio,
                                     note['onset'], note['pitch'])
                if pred < 0:
                    continue
                ds_total += 1
                if pred == note['string']:
                    ds_correct += 1
        
        acc = ds_correct / ds_total if ds_total > 0 else 0
        print("  %s: %d/%d = %.4f (%.2f%%)" % (ds_name, ds_correct, ds_total, acc, acc*100))
        by_dataset[ds_name] = (ds_correct, ds_total)
        all_correct += ds_correct
        all_total += ds_total
    
    print("\n=== IDMT-SMT-Guitar Cross-Dataset Results ===")
    overall = all_correct / all_total if all_total > 0 else 0
    print("Overall: %d/%d = %.4f (%.2f%%)" % (all_correct, all_total, overall, overall*100))
    for ds_name, (c, t) in by_dataset.items():
        acc = c / t if t > 0 else 0
        print("  %s: %d/%d = %.2f%%" % (ds_name, c, t, acc*100))

if __name__ == '__main__':
    main()
