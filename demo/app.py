"""Gradio public demo for morseformer.

Two input sources (upload + browser mic) + a preset selector + a text
output. Designed to run as a Hugging Face Space; the model is downloaded
from the Hub on first request and kept resident.

Run locally::

    pip install "morseformer[demo]"
    python demo/app.py

Deploy to a HF Space by pushing this repo to https://huggingface.co/spaces
and setting ``app.py`` as the entry point. The Space's requirements.txt
needs ``morseformer[demo]`` plus the CPU PyTorch wheel.
"""

from __future__ import annotations

import os
from functools import lru_cache
from math import gcd
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from scipy.signal import resample_poly

from morseformer.cli.presets import DEFAULT_PRESET, PRESETS
from morseformer.cli.registry import resolve_model
from morseformer.decoding.streaming import StreamingConfig, decode_offline
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.lm import GptLM, LmConfig
from morseformer.models.rnnt import RnntConfig, RnntModel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TARGET_SAMPLE_RATE = 8000


def _load_model(name: str) -> RnntModel:
    path = resolve_model(name)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    for k, v in (ckpt.get("ema") or {}).items():
        if k in state:
            state[k] = v
    enc = ckpt["config"]["model"]["encoder"]
    vocab_size = ckpt["config"]["model"].get("vocab_size")
    extra = {"vocab_size": vocab_size} if vocab_size is not None else {}
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"], **extra,
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=ckpt["config"]["model"]["d_pred"],
        pred_lstm_layers=ckpt["config"]["model"]["pred_lstm_layers"],
        d_joint=ckpt["config"]["model"]["d_joint"],
        dropout=ckpt["config"]["model"]["dropout"],
        **extra,
    )
    model = RnntModel(rnnt_cfg).to(_DEVICE).eval()
    model.load_state_dict(state)
    return model


def _load_lm(name: str) -> GptLM:
    path = resolve_model(name)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    for k, v in (ckpt.get("ema") or {}).items():
        if k in state:
            state[k] = v
    cfg = ckpt["config"]["model"]
    lm = GptLM(
        LmConfig(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
        )
    ).to(_DEVICE).eval()
    lm.load_state_dict(state)
    return lm


@lru_cache(maxsize=4)
def _get_model(name: str) -> RnntModel:
    return _load_model(name)


@lru_cache(maxsize=4)
def _get_lm(name: str) -> GptLM:
    return _load_lm(name)


def _to_mono_float32(audio_tuple: tuple[int, np.ndarray]) -> tuple[np.ndarray, int]:
    sample_rate, data = audio_tuple
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if arr.dtype.kind in ("i", "u"):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32)
    return arr, int(sample_rate)


def _resample(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == _TARGET_SAMPLE_RATE:
        return audio.astype(np.float32)
    g = gcd(src_rate, _TARGET_SAMPLE_RATE)
    return resample_poly(audio, _TARGET_SAMPLE_RATE // g, src_rate // g).astype(np.float32)


def decode(audio, preset_name: str) -> str:
    if audio is None:
        return "(no audio — record from your mic or upload a .wav)"
    arr, sr = _to_mono_float32(audio)
    if arr.size == 0:
        return "(empty audio)"
    arr = _resample(arr, sr)
    preset = PRESETS[preset_name]
    model = _get_model(preset.acoustic)
    cfg = StreamingConfig(
        confidence_threshold=preset.confidence_threshold,
        digit_threshold=preset.digit_threshold,
    )
    lm = _get_lm(preset.lm) if preset.lm and preset.fusion_weight > 0 else None
    text = decode_offline(
        model,
        arr,
        cfg,
        device=_DEVICE,
        lm=lm,
        fusion_weight=preset.fusion_weight if lm is not None else 0.0,
    )
    return text or "(decoder returned no text — try a louder / clearer sample)"


def _example_files() -> list[list]:
    candidates = [
        Path("test.wav"),
        Path("test_8k.wav"),
        Path("test_manu_8k.wav"),
    ]
    examples: list[list] = []
    for p in candidates:
        if p.exists():
            examples.append([str(p), DEFAULT_PRESET])
    return examples


def _preset_info() -> str:
    parts: list[str] = []
    for name in PRESETS:
        description = PRESETS[name].description.split(".")[0]
        if PRESETS[name].lm:
            description += "; first decode also downloads the LM"
        parts.append(f"{name}: {description}")
    return ", ".join(parts)


def build_app() -> gr.Blocks:
    preset_names = list(PRESETS.keys())
    with gr.Blocks(title="morseformer — open-source CW decoder") as app:
        gr.Markdown(
            "# morseformer\n"
            "Open-source Conformer + RNN-T Morse/CW decoder. "
            "Upload a `.wav` or record from your browser mic, pick a preset, "
            "and the model returns the decoded text.\n\n"
            "Note: the browser mic uses the OS-default device. To target a "
            "specific input (e.g. USB) launch the local PySide6 GUI with "
            "`morseformer gui`."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Audio input",
                )
                preset = gr.Dropdown(
                    choices=preset_names,
                    value=DEFAULT_PRESET,
                    label="Preset",
                    info=_preset_info(),
                )
                btn = gr.Button("Decode", variant="primary")
            with gr.Column():
                out = gr.Textbox(
                    label="Transcript",
                    lines=8,
                    interactive=False,
                    show_copy_button=True,
                )
        btn.click(decode, inputs=[audio_in, preset], outputs=out)

        examples = _example_files()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[audio_in, preset],
                outputs=out,
                fn=decode,
                cache_examples=False,
                label="Examples (click to try)",
            )
    return app


if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 7860))
    build_app().launch(server_name="0.0.0.0", server_port=server_port)
