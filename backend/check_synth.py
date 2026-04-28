"""SynthTabデータ構造確認"""
import pickle, glob, os

files = glob.glob(r'D:\Music\datasets\martin_finger\part_1_-_1_to_B_C(martin_finger)\**\*.pkl', recursive=True)[:5]
for f in files:
    bn = os.path.basename(f)
    data = pickle.load(open(f, 'rb'))
    if isinstance(data, dict):
        print(f'{bn}: dict keys={list(data.keys())[:10]}')
        for k, v in list(data.items())[:3]:
            if hasattr(v, 'shape'):
                print(f'  {k}: shape={v.shape}')
            elif isinstance(v, list):
                print(f'  {k}: list len={len(v)}, first={v[0] if v else None}')
            else:
                print(f'  {k}: {type(v).__name__}')
    elif hasattr(data, 'shape'):
        print(f'{bn}: ndarray shape={data.shape}, dtype={data.dtype}')
    else:
        tp = type(data).__name__
        ln = len(data) if hasattr(data, '__len__') else '?'
        print(f'{bn}: type={tp}, len={ln}')
