import mirdata
try:
    gs = mirdata.initialize('guitarset', data_home='D:/Music/datasets/GuitarSet')
    print(f'Total tracks found: {len(gs.track_ids)}')
    if len(gs.track_ids) > 0:
        print(f'First track ID format: {gs.track_ids[0]}')
        
    print('\nTrying full ID with _solo: 05_SS3-84-Bb_solo')
    try:
        t1 = gs.track('05_SS3-84-Bb_solo')
        print(f'SUCCESS: Found full ID!')
        print(f'  audio_mic_path: {getattr(t1, "audio_mic_path", "MISSING ATTRIBUTE")}')
        print(f'  audio_mix_path: {getattr(t1, "audio_mix_path", "MISSING ATTRIBUTE")}')
        import os
        if hasattr(t1, "audio_mic_path") and t1.audio_mic_path:
            print(f'  Exists on disk (mic)? {os.path.exists(t1.audio_mic_path)}')
    except Exception as e:
        print(f'FAIL: {type(e).__name__} - {e}')
        
    print('\nTrying stripped ID: 05_SS3-84-Bb')
    try:
        t2 = gs.track('05_SS3-84-Bb')
        print(f'SUCCESS: Found stripped ID! Audio path: {t2.audio_mic_path}')
    except Exception as e:
        print(f'FAIL: {type(e).__name__} - {e}')
        
except Exception as main_e:
    print(f'Main error: {main_e}')
