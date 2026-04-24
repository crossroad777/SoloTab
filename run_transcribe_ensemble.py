import sys, json, os

if __name__ == '__main__':
    sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')
    from ensemble_transcriber import transcribe_ensemble
    WAV = r'd:\Music\nextchord-solotab\uploads\20260320-000856-yt-052e93\converted.wav'
    tuning = [40, 45, 50, 55, 59, 64]
    
    print('Running transcribe_ensemble...')
    try:
        res = transcribe_ensemble(str(WAV), tuning_pitches=tuning)
        with open(r'd:\Music\nextchord-solotab\test_ens_raw.json', 'w') as f:
            json.dump(res['notes'], f)
        print('Done. Total notes:', len(res['notes']))
    except Exception as e:
        print("Error:", e)
