"""
train_gaps_multitask.py — GuitarSet + GAPS Multi-task学習
==============================================================
GuitarSet (スチール弦) と GAPS (ナイロン弦・クラシカルギター) を
混合してトレーニングし、onset/pitch検出の汎化能力を向上させる。

GAPSのMIDIからstring/fretアノテーションを自動生成し、
既存のCombinedLoss (onset + fret) で統一的に学習する。

Usage:
    # Step 1: GAPSデータの前処理（初回のみ・約30-60分）
    python train_gaps_multitask.py --preprocess-only

    # Step 2: Multi-task学習
    python train_gaps_multitask.py --domain martin_finger

    # 両方を一括実行
    python train_gaps_multitask.py --domain martin_finger --preprocess
"""
import os, sys, io, json, time, argparse, csv
import torch
import numpy as np
import librosa
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, Dataset as TorchDataset, ConcatDataset
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ["TQDM_ASCII"] = " 123456789#"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.batching import collate_fn_pad
from data_processing import dataset as gs_dataset
from data_processing.dataset import create_frame_level_labels
from training import loss_functions, epoch_processing
from model import architecture
import config

# ============================================================
# GAPS Dataset Configuration
# ============================================================
GAPS_DATA_DIR = r"D:\Music\datasets\GAPS_DATA"
GAPS_PROCESSED_DIR = os.path.join(GAPS_DATA_DIR, "_processed")
GAPS_METADATA_CSV = os.path.join(GAPS_DATA_DIR, "gaps_metadata_with_splits.csv")

# Guitar standard tuning (MIDI note numbers for open strings)
# String 0=Low E(40), 1=A(45), 2=D(50), 3=G(55), 4=B(59), 5=High E(64)
OPEN_STRING_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

OUTPUT_BASE_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")
TRAINING_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "training_output")

# AG-PT-set paths
AGPT_PROCESSED_DIR = os.path.join(r"D:\Music\datasets\AG-PT-set\aGPTset", "_processed")
AGPT_RATIO = 0.2   # AG-PT-set混合比率 (0.2: ~72件サンプリング)

# Synth V2 paths
SYNTH_V2_DIR = r"D:\Music\datasets\synth_v2"
SYNTH_V2_IDS = os.path.join(SYNTH_V2_DIR, "train_ids.txt")
SYNTH_RATIO = 0.5  # Synth V2混合比率 (0.5: 5000→~286件/エポック = GuitarSetと同数)

# Training Hyperparameters
FT_LR = 3e-6       # 低めのLR: 2ドメイン混合で安定学習
FT_EPOCHS = 9999
FT_PATIENCE = 10
FT_BATCH_SIZE = 1
GAPS_RATIO = 0.6   # GAPS混合比率 (0.6: 全371件使用)
ACCUMULATION_STEPS = 4  # Gradient Accumulation: 実効バッチサイズ = 1 * 4 = 4


# ============================================================
# MIDI → String/Fret 変換
# ============================================================
def pitch_to_string_fret(midi_pitch, max_fret=config.MAX_FRETS):
    """
    MIDIピッチをstring/fretに変換。最も低いフレット位置を優先。
    
    Returns:
        (string_idx, fret_num) or None if out of range
    """
    candidates = []
    for s_idx, open_pitch in OPEN_STRING_MIDI.items():
        fret = int(round(midi_pitch)) - open_pitch
        if 0 <= fret <= max_fret:
            candidates.append((s_idx, fret))
    if not candidates:
        return None
    # 最低フレット優先（最もナチュラルなポジション）
    return min(candidates, key=lambda x: x[1])


def parse_midi_to_raw_labels(midi_path, max_fret=config.MAX_FRETS):
    """
    MIDIファイルからraw_labelsテンソルを生成。
    
    Returns:
        torch.Tensor: shape [N, 5] - [onset_sec, offset_sec, string, fret, pitch]
        GuitarSet/SynthTabと完全互換のフォーマット
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi が必要です: pip install pretty_midi")
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes_list = []
    
    for instrument in pm.instruments:
        for note in instrument.notes:
            sf = pitch_to_string_fret(note.pitch, max_fret)
            if sf is None:
                continue  # ギター音域外のノートはスキップ
            string_idx, fret_num = sf
            notes_list.append([
                note.start,         # onset_sec
                note.end,           # offset_sec
                float(string_idx),  # string
                float(fret_num),    # fret
                float(note.pitch),  # pitch_midi
            ])
    
    if not notes_list:
        return torch.zeros((0, 5), dtype=torch.float32)
    
    labels = np.array(notes_list, dtype=np.float32)
    # onset順にソート
    labels = labels[labels[:, 0].argsort()]
    return torch.from_numpy(labels)


# ============================================================
# GAPS 前処理
# ============================================================
def preprocess_gaps_dataset(gaps_dir=GAPS_DATA_DIR, output_dir=GAPS_PROCESSED_DIR,
                            max_tracks=None):
    """
    GAPSデータセットの全トラックを前処理。
    WAV → CQT特徴量、MIDI → raw_labelsテンソル を保存。
    """
    audio_dir = os.path.join(gaps_dir, "audio")
    midi_dir = os.path.join(gaps_dir, "midi")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # メタデータCSVからsplit情報を取得
    splits = {}
    if os.path.exists(GAPS_METADATA_CSV):
        with open(GAPS_METADATA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get('id', '').strip()
                split = row.get('split', 'train').strip()
                if tid:
                    splits[tid] = split
    
    # WAVファイル一覧を取得
    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if max_tracks:
        wav_files = wav_files[:max_tracks]
    
    train_ids = []
    val_ids = []
    processed = 0
    skipped = 0
    errors = 0
    
    print(f"\n{'='*60}")
    print(f"  GAPS Preprocessing: {len(wav_files)} tracks")
    print(f"{'='*60}")
    
    for wav_path in tqdm(wav_files, desc="Processing GAPS", unit="track"):
        track_id = wav_path.stem  # e.g., "001_mvswc"
        feat_path = os.path.join(output_dir, f"{track_id}_features.pt")
        label_path = os.path.join(output_dir, f"{track_id}_labels.pt")
        
        # スキップ: 既に処理済み
        if os.path.exists(feat_path) and os.path.exists(label_path):
            split = splits.get(track_id, "train")
            if split == "test":
                val_ids.append(track_id)
            else:
                train_ids.append(track_id)
            skipped += 1
            continue
        
        # MIDI対応ファイルを探す
        midi_path = os.path.join(midi_dir, f"{track_id}.mid")
        if not os.path.exists(midi_path):
            errors += 1
            continue
        
        try:
            # CQT特徴量の計算
            audio, _ = librosa.load(str(wav_path), sr=config.SAMPLE_RATE, mono=True)
            cqt = librosa.cqt(
                y=audio,
                sr=config.SAMPLE_RATE,
                hop_length=config.HOP_LENGTH,
                fmin=config.FMIN_CQT,
                n_bins=config.N_BINS_CQT,
                bins_per_octave=config.BINS_PER_OCTAVE_CQT,
            )
            log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            features = torch.tensor(log_cqt, dtype=torch.float32)
            
            # MIDIからラベル生成
            raw_labels = parse_midi_to_raw_labels(midi_path, config.MAX_FRETS)
            
            if raw_labels.shape[0] == 0:
                errors += 1
                continue
            
            # 保存
            torch.save(features, feat_path)
            torch.save(raw_labels, label_path)
            
            split = splits.get(track_id, "train")
            if split == "test":
                val_ids.append(track_id)
            else:
                train_ids.append(track_id)
            processed += 1
            
        except Exception as e:
            print(f"\n  Error processing {track_id}: {e}")
            errors += 1
    
    # ID リスト保存
    with open(os.path.join(output_dir, "train_ids.txt"), "w") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(output_dir, "val_ids.txt"), "w") as f:
        f.write("\n".join(val_ids))
    
    print(f"\n  Processed: {processed}, Skipped (existing): {skipped}, Errors: {errors}")
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"  Output: {output_dir}")
    return train_ids, val_ids


# ============================================================
# GAPS Dataset Class
# ============================================================
class GAPSTabDataset(TorchDataset):
    """
    前処理済みGAPSデータのDataset。
    GuitarSetTabDatasetと同じ出力フォーマット:
    (features, labels_tuple, raw_labels, track_id)
    """
    def __init__(self, processed_dir, ids_file, hop_length, sr, max_fret):
        self.items = []
        self.hop_length = hop_length
        self.sr = sr
        self.max_fret = max_fret
        
        if not os.path.exists(ids_file):
            print(f"  [WARN] GAPS IDs file not found: {ids_file}")
            return
        
        with open(ids_file, 'r') as f:
            for line in f:
                tid = line.strip()
                if not tid:
                    continue
                feat_path = os.path.join(processed_dir, f"{tid}_features.pt")
                label_path = os.path.join(processed_dir, f"{tid}_labels.pt")
                if os.path.exists(feat_path) and os.path.exists(label_path):
                    self.items.append((feat_path, label_path, tid))
    
    def __len__(self):
        return len(self.items)
    
    # GAPSの長大サンプルをGuitarSet相当の長さに制限（速度改善）
    MAX_FRAMES = 3000  # GuitarSet最大≒約30秒相当

    def __getitem__(self, index):
        feat_path, label_path, tid = self.items[index]
        features = torch.load(feat_path, weights_only=False)
        raw_labels = torch.load(label_path, weights_only=False)

        # フレーム数キャップ: 長いサンプルはランダム位置で切り出し
        n_frames = features.shape[-1]
        if n_frames > self.MAX_FRAMES:
            import random
            start = random.randint(0, n_frames - self.MAX_FRAMES)
            features = features[..., start:start + self.MAX_FRAMES]
            # ラベルも対応する時間範囲にフィルタリング
            # raw_labels: Tensor [N, 5] = [onset_sec, offset_sec, string, fret, pitch]
            start_sec = start * self.hop_length / self.sr
            end_sec = (start + self.MAX_FRAMES) * self.hop_length / self.sr
            if isinstance(raw_labels, torch.Tensor) and raw_labels.ndim == 2 and raw_labels.shape[0] > 0:
                mask = (raw_labels[:, 0] >= start_sec) & (raw_labels[:, 0] < end_sec)
                raw_labels = raw_labels[mask].clone()
                raw_labels[:, 0] -= start_sec  # onset調整
                raw_labels[:, 1] = torch.clamp(raw_labels[:, 1], max=end_sec) - start_sec  # offset調整
            elif isinstance(raw_labels, torch.Tensor):
                pass  # 空テンソルはそのまま

        labels_tuple = create_frame_level_labels(
            raw_labels, features, self.hop_length, self.sr, self.max_fret
        )
        return features, labels_tuple, raw_labels, tid


# ============================================================
# Multi-task Training
# ============================================================
def finetune_multitask(domain, device, ft_epochs=FT_EPOCHS, ft_patience=FT_PATIENCE, include_agpt=False, ga_retrain=False, include_synth=False, include_idmt=False):
    ds_parts = ["GuitarSet", "GAPS"]
    if include_agpt: ds_parts.append("AG-PT")
    if include_synth: ds_parts.append("Synth")
    if include_idmt: ds_parts.append("IDMT")
    ds_label = " + ".join(ds_parts)
    print(f"\n{'='*60}")
    print(f"  Multi-task FT ({ds_label}): {domain}")
    print(f"{'='*60}")
    
    # ソースモデル選択
    if include_idmt:
        # IDMT混合: 3DS_GAモデルを初期重みとして使用、新suffix
        suffix = "multitask_4ds"
        src_ga = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_multitask_3ds_ga", "best_model.pth")
        src_3ds = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_multitask_3ds", "best_model.pth")
        src_path = src_ga if os.path.exists(src_ga) else (src_3ds if os.path.exists(src_3ds) else None)
    elif ga_retrain:
        # GA再学習: 3DSモデルを初期重みとして使用
        suffix = "multitask_3ds_ga"
        src_3ds = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_multitask_3ds", "best_model.pth")
        src_mt = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_multitask", "best_model.pth")
        src_path = src_3ds if os.path.exists(src_3ds) else (src_mt if os.path.exists(src_mt) else None)
    else:
        # 通常: GuitarSet FT済み or 元ドメインモデル
        suffix = "multitask_3ds" if include_agpt else "multitask"
        guitarset_ft_dir = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_guitarset_ft")
        domain_src_dir = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_model")
        src_candidates = [
            os.path.join(guitarset_ft_dir, "best_model.pth"),
            os.path.join(guitarset_ft_dir, "best_model_pre_emajor.pth"),
            os.path.join(domain_src_dir, "best_model.pth"),
        ]
        src_path = None
        for c in src_candidates:
            if os.path.exists(c):
                src_path = c
                break
    
    if src_path is None:
        print(f"[SKIP] No source model found for {domain}")
        return
    
    out_dir = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pth")
    log_path = os.path.join(out_dir, "training_log.txt")
    
    # ログからresume
    start_epoch = 1
    best_f1 = 0.0
    already_early_stopped = False
    if os.path.exists(log_path):
        import re
        with open(log_path, "r", encoding="utf-8") as lf:
            for line in lf:
                m = re.search(r"Epoch (\d+) \|.*F1: ([\d.]+)", line)
                if m:
                    start_epoch = int(m.group(1)) + 1
                m2 = re.search(r"Best F1: ([\d.]+)", line)
                if m2:
                    best_f1 = max(best_f1, float(m2.group(1)))
                if re.search(r"Early stop at epoch", line):
                    already_early_stopped = True
    if already_early_stopped:
        print(f"[SKIP] Already early-stopped for {domain} (Best F1={best_f1:.4f})")
        return
    if start_epoch > ft_epochs:
        print(f"[SKIP] Already completed for {domain}")
        return
    
    load_path = best_path if (start_epoch > 1 and os.path.exists(best_path)) else src_path
    print(f"  Loading from: {load_path} (epoch {start_epoch})")
    
    # ハイパーパラメータ（domain_src_dirは常に必要）
    domain_src_dir = os.path.join(TRAINING_OUTPUT_DIR, f"finetuned_{domain}_model")
    cfg_path = os.path.join(domain_src_dir, "run_configuration.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(TRAINING_OUTPUT_DIR, "baseline_model", "run_configuration.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        hp = json.load(f)["hyperparameters_tuned"]
    
    # --- DataLoaders ---
    common = config.DATASET_COMMON_PARAMS
    
    # GuitarSet train/val
    gs_train = gs_dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name="train",
        guitarset_data_home=config.DATA_HOME_DEFAULT,
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_TRAIN_AUGMENTATION_PARAMS,  # 拡張有効: ピッチシフト/タイムストレッチ/ノイズ/リバーブ/EQ/SpecAugment
    )
    gs_val = gs_dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name="validation",
        label_transform_function=create_frame_level_labels,
        **common, **config.DATASET_EVAL_AUGMENTATION_PARAMS,
    )
    
    # GAPS train
    gaps_train = GAPSTabDataset(
        processed_dir=GAPS_PROCESSED_DIR,
        ids_file=os.path.join(GAPS_PROCESSED_DIR, "train_ids.txt"),
        hop_length=config.HOP_LENGTH, sr=config.SAMPLE_RATE,
        max_fret=common.get('max_fret_value', config.MAX_FRETS),
    )
    
    # E major合成データ
    emajor_ids_path = os.path.join(OUTPUT_BASE_DIR, "emajor_train_ids.txt")
    extra_datasets = []
    if len(gaps_train) > 0:
        # GAPS混合比率に基づいてサブサンプリング
        target_gaps_size = int(len(gs_train) * GAPS_RATIO / (1 - GAPS_RATIO))
        if target_gaps_size < len(gaps_train):
            indices = torch.randperm(len(gaps_train))[:target_gaps_size].tolist()
            gaps_subset = torch.utils.data.Subset(gaps_train, indices)
            extra_datasets.append(gaps_subset)
            print(f"  GAPS: {len(gaps_train)} total, {target_gaps_size} sampled (ratio={GAPS_RATIO})")
        else:
            extra_datasets.append(gaps_train)
            print(f"  GAPS: {len(gaps_train)} (all)")
    else:
        print("  [WARN] No GAPS data found. Run with --preprocess first.")
    
    # Inline SynthDataset (emajor / synth_v2 共用)
    class _SynthDS(TorchDataset):
        def __init__(self, base_dir, ids_file, hop_length, sr, max_fret):
            self.items = []
            self.hop_length = hop_length
            self.sr = sr
            self.max_fret = max_fret
            with open(ids_file, 'r') as f:
                for line in f:
                    tid = line.strip()
                    if not tid: continue
                    fp = os.path.join(base_dir, f"{tid}_features.pt")
                    lp = os.path.join(base_dir, f"{tid}_labels.pt")
                    if os.path.exists(fp) and os.path.exists(lp):
                        self.items.append((fp, lp, tid))
        def __len__(self): return len(self.items)
        def __getitem__(self, index):
            fp, lp, tid = self.items[index]
            features = torch.load(fp, weights_only=False)
            raw_labels = torch.load(lp, weights_only=False)
            labels_tuple = create_frame_level_labels(
                raw_labels, features, self.hop_length, self.sr, self.max_fret)
            return features, labels_tuple, raw_labels, tid
    
    # E major synth — 合成音+調性バイアスのリスクがあるため無効化
    # if os.path.exists(emajor_ids_path):
    #     emajor_ds = _SynthDS(
    #         os.path.join(OUTPUT_BASE_DIR, "train"), emajor_ids_path,
    #         config.HOP_LENGTH, config.SAMPLE_RATE,
    #         common.get('max_fret_value', config.MAX_FRETS),
    #     )
    #     if len(emajor_ds) > 0:
    #         extra_datasets.append(emajor_ds)
    #         print(f"  E major synth: {len(emajor_ds)}")
    
    # Synth V2 (optional)
    if include_synth and os.path.exists(SYNTH_V2_IDS):
        synth_v2_ds = _SynthDS(
            SYNTH_V2_DIR, SYNTH_V2_IDS,
            config.HOP_LENGTH, config.SAMPLE_RATE,
            common.get('max_fret_value', config.MAX_FRETS),
        )
        if len(synth_v2_ds) > 0:
            target_synth_size = int(len(gs_train) * SYNTH_RATIO / (1 - SYNTH_RATIO))
            if target_synth_size < len(synth_v2_ds):
                synth_indices = torch.randperm(len(synth_v2_ds))[:target_synth_size].tolist()
                synth_subset = torch.utils.data.Subset(synth_v2_ds, synth_indices)
                extra_datasets.append(synth_subset)
                print(f"  Synth V2: {len(synth_v2_ds)} total, {target_synth_size} sampled (ratio={SYNTH_RATIO})")
            else:
                extra_datasets.append(synth_v2_ds)
                print(f"  Synth V2: {len(synth_v2_ds)} (all)")
    elif include_synth:
        print("  [WARN] Synth V2 not found. Run generate_dataset.py first.")
    
    # IDMT-SMT-V2 (optional)
    if include_idmt:
        idmt_dir = r"D:\Music\datasets\idmt_processed"
        idmt_ids = os.path.join(idmt_dir, "train_ids.txt")
        if os.path.exists(idmt_ids):
            idmt_ds = _SynthDS(
                idmt_dir, idmt_ids,
                config.HOP_LENGTH, config.SAMPLE_RATE,
                common.get('max_fret_value', config.MAX_FRETS),
            )
            if len(idmt_ds) > 0:
                extra_datasets.append(idmt_ds)
                print(f"  IDMT-SMT: {len(idmt_ds)}")
        else:
            print("  [WARN] IDMT-SMT not found. Run preprocess_idmt.py first.")
    
    # AG-PT-set (optional)
    if include_agpt:
        agpt_ids_path = os.path.join(AGPT_PROCESSED_DIR, "train_ids.txt")
        if os.path.exists(agpt_ids_path):
            agpt_ds = GAPSTabDataset(
                processed_dir=AGPT_PROCESSED_DIR,
                ids_file=agpt_ids_path,
                hop_length=config.HOP_LENGTH, sr=config.SAMPLE_RATE,
                max_fret=common.get('max_fret_value', config.MAX_FRETS),
            )
            if len(agpt_ds) > 0:
                target_agpt_size = int(len(gs_train) * AGPT_RATIO / (1 - AGPT_RATIO))
                if target_agpt_size < len(agpt_ds):
                    agpt_indices = torch.randperm(len(agpt_ds))[:target_agpt_size].tolist()
                    agpt_subset = torch.utils.data.Subset(agpt_ds, agpt_indices)
                    extra_datasets.append(agpt_subset)
                    print(f"  AG-PT: {len(agpt_ds)} total, {target_agpt_size} sampled (ratio={AGPT_RATIO})")
                else:
                    extra_datasets.append(agpt_ds)
                    print(f"  AG-PT: {len(agpt_ds)} (all)")
        else:
            print("  [WARN] AG-PT-set not found. Run preprocess_agpt.py first.")

    # Combined dataset
    all_train = [gs_train] + extra_datasets
    combined_train = ConcatDataset(all_train) if len(all_train) > 1 else gs_train
    
    train_loader = DataLoader(combined_train, batch_size=FT_BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_pad, num_workers=0, pin_memory=True)
    val_loader = DataLoader(gs_val, batch_size=FT_BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_pad, num_workers=0, pin_memory=True)
    print(f"  Combined Train: {len(combined_train)}, Val (GuitarSet only): {len(gs_val)}")
    
    # --- Model ---
    with torch.no_grad():
        cnn_out_dim = architecture.TabCNN()(torch.randn(1, 1, config.N_BINS_CQT, 32))
        cnn_out_dim = cnn_out_dim.shape[1] * cnn_out_dim.shape[2]
    
    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hp.get("RNN_TYPE", "GRU"),
        rnn_hidden_size=hp["RNN_HIDDEN_SIZE"],
        rnn_layers=hp["RNN_LAYERS"],
        rnn_dropout=hp["RNN_DROPOUT"],
        rnn_bidirectional=hp.get("RNN_BIDIRECTIONAL", True),
    )
    sd = torch.load(load_path, map_location=device, weights_only=False)
    if list(sd.keys())[0].startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    
    onset_pw = (torch.tensor([hp["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device)
                if hp.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0 else None)
    criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pw, onset_loss_weight=hp["ONSET_LOSS_WEIGHT"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=hp["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=7)
    
    
    # --- AMP (Mixed Precision) --- disabled: batch_size=1では品質劣化が確認されたため無効
    scaler = None
    
    # --- Training Loop ---
    bad_epochs = 0
    mode = "a" if start_epoch > 1 else "w"
    with open(log_path, mode, encoding="utf-8") as log_f:
        if mode == "w":
            log_f.write(f"--- Multi-task FT (GuitarSet+GAPS) for {domain} | {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_f.write(f"    GuitarSet: {len(gs_train)}, GAPS: {sum(len(d) for d in extra_datasets)}\n")
        
        for epoch in range(start_epoch, ft_epochs + 1):
            desc = f"Epoch {epoch}/{ft_epochs}"
            
            train_pbar = tqdm(train_loader, desc=f"{desc} [Train]", unit="b",
                              leave=False, dynamic_ncols=True)
            accum = ACCUMULATION_STEPS if ga_retrain else 1
            train_m = epoch_processing.train_one_epoch(model, train_pbar, optimizer,
                                                        criterion, device,
                                                        scaler=scaler,
                                                        accumulation_steps=accum)
            
            val_pbar = tqdm(val_loader, desc=f"{desc} [Val]", unit="b",
                            leave=False, dynamic_ncols=True)
            val_m = epoch_processing.evaluate_one_epoch(model, val_pbar, criterion,
                                                        device, config)
            
            f1 = val_m.get("val_tdr_f1_at_0.5", 0.0)
            p = val_m.get("val_tdr_precision_at_0.5", 0.0)
            r = val_m.get("val_tdr_recall_at_0.5", 0.0)
            scheduler.step(f1)
            
            line = f"Epoch {epoch} | Loss: {train_m['train_total_loss']:.4f} | F1: {f1:.4f} P: {p:.4f} R: {r:.4f}"
            print(line)
            log_f.write(line + "\n")
            
            if f1 >= best_f1:
                best_f1 = f1
                bad_epochs = 0
                torch.save(model.state_dict(), best_path)
                msg = f"  -> Best F1: {best_f1:.4f} saved!"
                print(msg)
                log_f.write(msg + "\n")
            else:
                bad_epochs += 1
                if bad_epochs >= ft_patience:
                    print(f"  Early stop at epoch {epoch}")
                    log_f.write(f"  Early stop at epoch {epoch}\n")
                    break
            log_f.flush()
    
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    print(f"  Completed {domain}: Best F1 = {best_f1:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GuitarSet + GAPS Multi-task Training")
    parser.add_argument("--domain", type=str, default="martin_finger",
                        help="Domain to fine-tune (default: martin_finger)")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess GAPS before training")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only preprocess GAPS (no training)")
    parser.add_argument("--epochs", type=int, default=FT_EPOCHS,
                        help=f"Max epochs (default: {FT_EPOCHS})")
    parser.add_argument("--patience", type=int, default=FT_PATIENCE,
                        help=f"Early stopping patience (default: {FT_PATIENCE})")
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Max GAPS tracks to preprocess (for testing)")
    parser.add_argument("--include-agpt", action="store_true",
                        help="Include AG-PT-set as 3rd dataset (3-dataset integration)")
    parser.add_argument("--include-synth", action="store_true",
                        help="Include Synth V2 as 4th dataset (100%% accurate labels)")
    parser.add_argument("--ga-retrain", action="store_true",
                        help="Gradient Accumulation retrain from 3DS models (accum_steps=4)")
    parser.add_argument("--include-idmt", action="store_true",
                        help="Include IDMT-SMT-V2 dataset (electric guitar, 252 tracks)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Step 1: GAPS前処理
    if args.preprocess or args.preprocess_only:
        preprocess_gaps_dataset(max_tracks=args.max_tracks)
        if args.preprocess_only:
            print("\nPreprocessing complete. Run without --preprocess-only to train.")
            return
    
    # Step 2: 前処理済みGAPSデータの存在確認
    gaps_ids = os.path.join(GAPS_PROCESSED_DIR, "train_ids.txt")
    if not os.path.exists(gaps_ids):
        print("[ERROR] GAPS preprocessed data not found.")
        print("        Run: python train_gaps_multitask.py --preprocess-only")
        return
    
    # Step 3: Multi-task学習
    finetune_multitask(args.domain, device, ft_epochs=args.epochs,
                       ft_patience=args.patience, include_agpt=args.include_agpt,
                       ga_retrain=args.ga_retrain, include_synth=args.include_synth,
                       include_idmt=args.include_idmt)
    
    print("\n" + "=" * 60)
    print("Multi-task training complete!")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
