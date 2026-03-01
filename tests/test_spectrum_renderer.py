from __future__ import annotations

import numpy as np

from spectrumov.renderer import (
    ReSpectrumRenderer,
    SpectrumRenderConfig,
    choose_audio_codec_for_container,
)


def test_render_frame_shape_and_dtype() -> None:
    cfg = SpectrumRenderConfig(width=320, height=120, fps=30.0, fft_size=2048)
    renderer = ReSpectrumRenderer(sample_rate=44100.0, config=cfg)

    block = np.zeros(cfg.fft_size, dtype=np.float32)
    frame = renderer.render_frame(block)

    assert frame.shape == (cfg.height, cfg.width, 3)
    assert frame.dtype == np.uint16


def test_grid_default_is_hidden() -> None:
    cfg = SpectrumRenderConfig()
    assert cfg.show_grid is False
    assert cfg.encoder == "ffmpeg"
    assert cfg.ffmpeg_vcodec == "prores_ks"
    assert cfg.render_bit_depth == 16
    assert cfg.curve_smoothing_sigma == 1.2


def test_audio_codec_selected_from_container() -> None:
    assert choose_audio_codec_for_container("out.mp4") == "aac"
    assert choose_audio_codec_for_container("out.mov") == "pcm_s16le"


def test_envelope_holds_then_decays() -> None:
    cfg = SpectrumRenderConfig(width=320, height=120, fps=60.0, fft_size=2048)
    renderer = ReSpectrumRenderer(sample_rate=44100.0, config=cfg)

    t = np.arange(cfg.fft_size, dtype=np.float64) / 44100.0
    tone_block = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    silence_block = np.zeros(cfg.fft_size, dtype=np.float32)

    curve_loud, _ = renderer.analyze_frame(tone_block)
    curve_after_silence, _ = renderer.analyze_frame(silence_block)

    idx = int(np.argmin(np.abs(renderer.group_freqs - 440.0)))

    # Higher amplitude means a smaller y value (closer to the top).
    assert curve_loud[idx] < renderer.bottom
    # After one silent frame, it should not instantly drop to floor.
    assert curve_after_silence[idx] < renderer.bottom


def test_show_peaks_produces_peak_curve() -> None:
    cfg = SpectrumRenderConfig(width=320, height=120, fps=30.0, fft_size=2048, show_peaks=True)
    renderer = ReSpectrumRenderer(sample_rate=44100.0, config=cfg)

    block = np.zeros(cfg.fft_size, dtype=np.float32)
    _, peak_y = renderer.analyze_frame(block)

    assert peak_y is not None
    assert peak_y.shape[0] == renderer.group_count
