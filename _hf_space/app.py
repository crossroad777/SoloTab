"""
SoloTab V2.0 — Pure MoE Guitar Tablature Transcriber
=====================================================
Hugging Face Spaces 用 Gradio アプリケーション
6つのドメイン特化CRNNエキスパートモデルによるアンサンブル推論
"""

import os
import tempfile
import numpy as np
import torch
import librosa
from scipy import stats
from huggingface_hub import hf_hub_download
import gradio as gr

import config
from model.architecture import GuitarTabCRNN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERT_NAMES = [
    "finetuned_martin_finger_guitarset_ft",
    "finetuned_taylor_finger_guitarset_ft",
    "finetuned_luthier_finger_guitarset_ft",
    "finetuned_martin_pick_guitarset_ft",
    "finetuned_taylor_pick_guitarset_ft",
    "finetuned_luthier_pick_guitarset_ft",
]
EXPERT_DISPLAY_NAMES = [
    "Martin Finger", "Taylor Finger", "Luthier Finger",
    "Martin Pick", "Taylor Pick", "Luthier Pick",
]

# TODO: ユーザーのHFリポジトリ名に変更してください
HF_MODEL_REPO = "Crossroad777/solotab-moe-models"

MAX_FRETS = config.MAX_FRETS
FRET_SILENCE_CLASS = MAX_FRETS + 1
OPEN_STRING_PITCHES = config.OPEN_STRING_PITCHES
HOP_LENGTH = config.HOP_LENGTH
SAMPLE_RATE = config.SAMPLE_RATE

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]  # 1弦→6弦

# ---------------------------------------------------------------------------
# Model Loading (cached at module level)
# ---------------------------------------------------------------------------
_models = []


def _download_and_load_models():
    """HF Hub からモデルをダウンロードしてロードする（初回のみ）"""
    global _models
    if _models:
        return _models

    device = torch.device("cpu")
    loaded = []

    for name in EXPERT_NAMES:
        filename = f"{name}/best_model.pth"
        print(f"Downloading {filename} ...")

        try:
            local_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=filename,
                cache_dir="/tmp/hf_cache",
            )
        except Exception as e:
            print(f"  -> Failed to download {filename}: {e}")
            # フォールバック: ローカルパスを試す（開発用）
            local_path = os.path.join("models", name, "best_model.pth")
            if not os.path.exists(local_path):
                print(f"  -> Local fallback also not found. Skipping.")
                continue

        print(f"Loading model from {local_path} ...")
        model = GuitarTabCRNN(
            num_frames_rnn_input_dim=1280,
            rnn_type="GRU",
            rnn_hidden_size=768,
            rnn_layers=2,
            rnn_dropout=0.3,
            rnn_bidirectional=True,
        )
        state_dict = torch.load(local_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded.append(model)
        print(f"  -> Loaded successfully.")

    _models = loaded
    print(f"Total models loaded: {len(_models)}")
    return _models


# ---------------------------------------------------------------------------
# Core Inference
# ---------------------------------------------------------------------------
def _extract_cqt(audio_path: str) -> torch.Tensor:
    """CQT特徴量を抽出する"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    cqt = librosa.cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
        bins_per_octave=config.BINS_PER_OCTAVE_CQT,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def _frames_to_notes(
    onset_probs: np.ndarray,
    fret_indices: np.ndarray,
    onset_threshold: float = 0.8,
) -> list[dict]:
    """フレーム単位の予測をノートリストに変換"""
    time_per_frame = HOP_LENGTH / SAMPLE_RATE
    num_frames, num_strings = onset_probs.shape
    notes = []

    for string_idx in range(num_strings):
        probs = onset_probs[:, string_idx]
        onsets = []
        for f in range(num_frames):
            p = probs[f]
            if p <= onset_threshold:
                continue
            left_ok = (f == 0) or (p > probs[f - 1])
            right_ok = (f == num_frames - 1) or (p >= probs[f + 1])
            if left_ok and right_ok:
                if not onsets or f != onsets[-1]:
                    onsets.append(f)

        for i, onset_frame in enumerate(onsets):
            fret_val = int(fret_indices[onset_frame, string_idx])
            if fret_val == FRET_SILENCE_CLASS or not (0 <= fret_val <= MAX_FRETS):
                for offset in (-1, 1, -2, 2):
                    fi = onset_frame + offset
                    if 0 <= fi < num_frames:
                        candidate = int(fret_indices[fi, string_idx])
                        if candidate != FRET_SILENCE_CLASS and 0 <= candidate <= MAX_FRETS:
                            fret_val = candidate
                            break
            if fret_val == FRET_SILENCE_CLASS or not (0 <= fret_val <= MAX_FRETS):
                continue

            next_onset = onsets[i + 1] if i + 1 < len(onsets) else num_frames
            max_dur = int(1.5 * SAMPLE_RATE / HOP_LENGTH)
            end_frame = min(onset_frame + max_dur, next_onset)
            musicxml_string = 6 - string_idx
            pitch = OPEN_STRING_PITCHES[string_idx] + fret_val

            notes.append({
                "start": round(onset_frame * time_per_frame, 4),
                "end": round(end_frame * time_per_frame, 4),
                "pitch": int(pitch),
                "string": musicxml_string,
                "fret": fret_val,
                "velocity": round(min(0.5 + float(probs[onset_frame]) * 0.5, 1.0), 4),
            })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def _transcribe_moe(
    audio_path: str,
    vote_threshold: int = 5,
    onset_threshold: float = 0.8,
    vote_prob_threshold: float = 0.5,
) -> list[dict]:
    """Pure MoE 推論"""
    models = _download_and_load_models()
    if not models:
        return []

    print("Extracting CQT features...")
    features = _extract_cqt(audio_path)

    all_onset_probs = []
    all_fret_preds = []

    for i, model in enumerate(models):
        print(f"Expert {i+1}/{len(models)} ({EXPERT_DISPLAY_NAMES[i]}) を推論中...")

        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()

        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))

    print("多数決投票中...")
    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)

    binary_votes = all_onset_probs > vote_prob_threshold
    vote_counts = np.sum(binary_votes, axis=0)
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < vote_threshold] = 0.0
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)

    print("ノートをデコード中...")
    notes = _frames_to_notes(consensus_onset_probs, consensus_frets, onset_threshold)

    for n in notes:
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])

    return notes


# ---------------------------------------------------------------------------
# Tablature Rendering
# ---------------------------------------------------------------------------
MIDI_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_note_name(midi: int) -> str:
    return f"{MIDI_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _render_tablature(notes: list[dict], duration: float) -> str:
    """ノートリストからテキストタブ譜を生成"""
    if not notes:
        return "（ノートが検出されませんでした）"

    time_per_slot = 0.15  # 秒/スロット
    total_slots = int(duration / time_per_slot) + 1
    slots_per_line = 60

    # 各弦のスロット配列を初期化
    tab_lines = {s: ["-"] * total_slots for s in range(6)}

    for note in notes:
        slot = int(note["start"] / time_per_slot)
        string_idx = 6 - note["string"]  # MusicXML弦番号 → 内部インデックス
        if 0 <= slot < total_slots and 0 <= string_idx < 6:
            fret_str = str(note["fret"])
            tab_lines[string_idx][slot] = fret_str

    # テキスト出力
    output_parts = []
    for start in range(0, total_slots, slots_per_line):
        end = min(start + slots_per_line, total_slots)
        time_label = f"[{start * time_per_slot:.1f}s]"
        output_parts.append(time_label)
        for s in range(6):  # e, B, G, D, A, E
            line_name = STRING_NAMES[s]
            slots = tab_lines[s][start:end]
            line_content = ""
            for slot_val in slots:
                if len(slot_val) == 1:
                    line_content += slot_val + "-"
                else:
                    line_content += slot_val
            output_parts.append(f"{line_name}|{line_content}|")
        output_parts.append("")

    return "\n".join(output_parts)


def _notes_to_midi(notes: list[dict], output_path: str):
    """ノートリストからMIDIファイルを生成（midiutil使用、なければスキップ）"""
    try:
        from midiutil import MIDIFile
    except ImportError:
        return None

    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    midi.addProgramChange(0, 0, 0, 25)  # Acoustic Guitar (steel)

    for note in notes:
        start_beat = note["start"] * 2  # 120BPM: 1秒 = 2拍
        duration_beat = max((note["end"] - note["start"]) * 2, 0.25)
        velocity = int(note["velocity"] * 127)
        midi.addNote(0, 0, note["pitch"], start_beat, duration_beat, velocity)

    with open(output_path, "wb") as f:
        midi.writeFile(f)
    return output_path


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
def transcribe(audio_path, vote_threshold, onset_threshold):
    """メイン推論関数（Gradioから呼ばれる）"""
    if audio_path is None:
        return "オーディオファイルをアップロードしてください。", "", None

    try:
        # 音声の長さを取得
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(y) / sr

        # MoE推論
        notes = _transcribe_moe(
            audio_path,
            vote_threshold=int(vote_threshold),
            onset_threshold=float(onset_threshold),
        )

        # サマリー
        summary_lines = [
            f"🎸 **SoloTab V2.0 — Pure MoE Transcription Result**",
            f"",
            f"- 検出ノート数: **{len(notes)}**",
            f"- 楽曲長: **{duration:.1f}秒**",
            f"- ノート密度: **{len(notes)/duration:.1f} notes/sec**" if duration > 0 else "",
            f"- 投票閾値: {int(vote_threshold)}/6, オンセット閾値: {onset_threshold:.2f}",
        ]

        if notes:
            from collections import Counter
            string_dist = Counter(n["string"] for n in notes)
            frets = [n["fret"] for n in notes]
            summary_lines.extend([
                f"- フレット範囲: {min(frets)}〜{max(frets)}",
                f"- 弦分布: " + ", ".join(f"{k}弦:{v}" for k, v in sorted(string_dist.items())),
            ])

        summary = "\n".join(summary_lines)

        # タブ譜
        tablature = _render_tablature(notes, duration)

        # MIDI出力
        midi_path = None
        if notes:
            midi_path = os.path.join(tempfile.gettempdir(), "solotab_output.mid")
            _notes_to_midi(notes, midi_path)

        return summary, tablature, midi_path

    except Exception as e:
        import traceback
        error_msg = f"❌ エラーが発生しました:\n\n```\n{traceback.format_exc()}\n```"
        return error_msg, "", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
# 🎸 SoloTab V2.0 — Pure MoE Guitar Tablature Transcriber

ソロギター音源をアップロードすると、AIが自動でタブ譜を生成します。

**仕組み:** 6つのドメイン特化CRNNモデル（Martin/Taylor/Luthier × Finger/Pick）が
独立して解析し、多数決で高精度なノート検出を行います。

**性能:** GuitarSet Test F1 = 0.8310 | Full F1 = 0.8478

> ⚠️ CPU推論のため、1曲あたり1〜3分程度かかる場合があります。
"""

with gr.Blocks(
    title="SoloTab V2.0 — Guitar Tablature Transcriber",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="indigo",
    ),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎵 ギター音源をアップロード",
                type="filepath",
                sources=["upload", "microphone"],
            )
            with gr.Row():
                vote_slider = gr.Slider(
                    minimum=1, maximum=6, step=1, value=5,
                    label="投票閾値（何モデル以上の合意が必要か）",
                )
                onset_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, step=0.05, value=0.8,
                    label="オンセット閾値",
                )
            transcribe_btn = gr.Button("🎼 タブ譜を生成", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="解析結果サマリー")
            tab_output = gr.Code(
                label="📋 タブ譜",
                language=None,
                lines=20,
            )
            midi_output = gr.File(label="📥 MIDIファイル ダウンロード")

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, vote_slider, onset_slider],
        outputs=[summary_output, tab_output, midi_output],
    )

if __name__ == "__main__":
    # ローカルテスト用
    demo.launch(server_name="0.0.0.0", server_port=7860)
