from __future__ import annotations

import numpy as np

from spectrumov.utils import (
    compute_decay_delta,
    compute_group_map,
    freq_to_x,
    jsfx_window,
    magnitude_to_01,
    normalize_audio_array,
    one_to_y,
    sum_channels_to_mono,
    x_to_freq,
)


def test_normalize_int16_audio() -> None:
    raw = np.array([-32768, 0, 32767], dtype=np.int16)
    norm = normalize_audio_array(raw)

    assert norm.dtype == np.float32
    assert np.isclose(norm[0], -1.0, atol=1e-6)
    assert np.isclose(norm[2], 32767 / 32768, atol=1e-6)


def test_sum_channels_to_mono_uses_sum_not_average() -> None:
    stereo = np.array([[0.25, -0.10], [0.20, 0.30]], dtype=np.float32)
    mono = sum_channels_to_mono(stereo)

    assert mono.shape == (2,)
    assert np.allclose(mono, np.array([0.15, 0.50], dtype=np.float32), atol=1e-7)


def test_jsfx_blackman_harris_window_is_symmetric_and_normalized() -> None:
    fft_size = 8192
    window = jsfx_window(window_type=2, fft_size=fft_size).astype(np.float64)

    assert window.shape == (fft_size,)
    assert np.all(window >= 0)
    assert np.allclose(window[1:], window[:0:-1], atol=1e-8)

    half = fft_size // 2
    reconstructed = np.sum(window[: half + 1]) * 2.0 - window[half]
    assert np.isclose(reconstructed, 0.5, rtol=1e-6)


def test_freq_to_x_roundtrip() -> None:
    freqs = np.array([10.0, 100.0, 1000.0, 10000.0, 22050.0])
    x = freq_to_x(freqs, width=1920, left_margin=0.0, right_margin=0.0, min_freq=10.0, max_freq=22050.0)
    restored = x_to_freq(
        x,
        width=1920,
        left_margin=0.0,
        right_margin=0.0,
        min_freq=10.0,
        max_freq=22050.0,
    )

    assert np.allclose(restored, freqs, rtol=1e-8)


def test_tilt_is_zero_at_1024_hz_anchor() -> None:
    magnitude = np.array([1e-4], dtype=np.float64)
    freq = np.array([1024.0], dtype=np.float64)

    z_no_tilt = magnitude_to_01(magnitude, freq, ceiling_db=0.0, noise_floor_db=-90.0, tilt_db_per_oct=0.0)
    z_tilted = magnitude_to_01(magnitude, freq, ceiling_db=0.0, noise_floor_db=-90.0, tilt_db_per_oct=4.5)

    assert np.allclose(z_no_tilt, z_tilted, atol=1e-12)


def test_group_map_is_monotonic() -> None:
    _, group_ids, group_freqs = compute_group_map(sample_rate=44100.0, fft_size=8192)

    assert np.all(np.diff(group_ids) >= 0)
    assert np.all(np.diff(group_freqs) > 0)


def test_decay_delta_range() -> None:
    delta_30 = compute_decay_delta(30.0)
    delta_60 = compute_decay_delta(60.0)

    assert 0.0 < delta_30 < 1.0
    assert 0.0 < delta_60 < 1.0
    assert delta_60 > delta_30


def test_one_to_y_mapping() -> None:
    z = np.array([0.0, 0.5, 1.0])
    y = one_to_y(z, height=500, top_margin=10.0, bottom_margin=0.0, text_height=12.0)

    assert np.isclose(y[0], 488.0)
    assert np.isclose(y[2], 10.0)
