from __future__ import annotations

import math

import numpy as np

SMALL_MAG = 1e-26


def normalize_audio_array(samples: np.ndarray) -> np.ndarray:
    """Convert WAV samples to float32 in a Reaper-like -1..1 scale."""
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)

    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        return samples.astype(np.float32) / scale

    raise TypeError(f"Unsupported audio dtype: {samples.dtype}")


def sum_channels_to_mono(samples: np.ndarray) -> np.ndarray:
    """Sum all channels to mono (not average), matching the requested behavior."""
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)

    if samples.ndim != 2:
        raise ValueError(f"Expected mono or multichannel audio, got shape={samples.shape}")

    return np.sum(samples.astype(np.float32, copy=False), axis=1)


def jsfx_window(window_type: int, fft_size: int) -> np.ndarray:
    """Build the FFT window using the same formulas/normalization as spectrum.jsfx-inc."""
    if fft_size <= 0 or fft_size % 2 != 0:
        raise ValueError("fft_size must be a positive even integer")

    half = fft_size // 2
    i = np.arange(half + 1, dtype=np.float64)
    windowpos = i * (2.0 * np.pi / fft_size)

    if window_type == 1:
        half_window = 0.53836 - np.cos(windowpos) * 0.46164
    elif window_type == 2:
        half_window = (
            0.35875
            - 0.48829 * np.cos(windowpos)
            + 0.14128 * np.cos(2.0 * windowpos)
            - 0.01168 * np.cos(3.0 * windowpos)
        )
    elif window_type == 3:
        half_window = 0.42 - 0.50 * np.cos(windowpos) + 0.08 * np.cos(2.0 * windowpos)
    elif window_type == 4:
        half_window = np.ones_like(windowpos)
    else:
        raise ValueError(f"Unsupported window_type={window_type}; expected 1..4")

    # Same normalization used in the JSFX code.
    pwr = 0.5 / (np.sum(half_window) * 2.0 - half_window[-1])
    half_window *= pwr

    window = np.empty(fft_size, dtype=np.float64)
    window[: half + 1] = half_window
    window[half + 1 :] = half_window[-2:0:-1]

    return window.astype(np.float32)


def compute_group_map(sample_rate: float, fft_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map linear FFT bins to the JSFX log-quantized bins used for drawing."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    half = fft_size // 2
    fft_freqs = np.arange(1, half + 1, dtype=np.float64) * (sample_rate / fft_size)
    quantized_bins = np.floor(32.0 * np.log(fft_freqs) / np.log(2.0)).astype(np.int64)

    changed = np.empty_like(quantized_bins, dtype=bool)
    changed[0] = True
    changed[1:] = quantized_bins[1:] != quantized_bins[:-1]

    group_ids = np.cumsum(changed) - 1
    group_count = int(group_ids[-1] + 1)

    # JSFX stores the dominant frequency as the latest frequency in each group.
    group_freqs = np.zeros(group_count, dtype=np.float64)
    group_freqs[group_ids] = fft_freqs

    return fft_freqs, group_ids, group_freqs


def reduce_fft_to_groups(power_bins: np.ndarray, group_ids: np.ndarray, group_count: int) -> np.ndarray:
    """Take max power per JSFX group (same behavior as repeated max assignment in JSFX)."""
    grouped = np.full(group_count, SMALL_MAG, dtype=np.float64)
    np.maximum.at(grouped, group_ids, power_bins)
    return grouped


def freq_to_x(
    freq: np.ndarray | float,
    width: int,
    left_margin: float,
    right_margin: float,
    min_freq: float,
    max_freq: float,
) -> np.ndarray:
    usable_width = width - left_margin - right_margin
    freq_log_max = math.log(max_freq / min_freq)
    freq_arr = np.asarray(freq, dtype=np.float64)
    return left_margin + usable_width * np.log(freq_arr / min_freq) / freq_log_max


def x_to_freq(
    x: np.ndarray | float,
    width: int,
    left_margin: float,
    right_margin: float,
    min_freq: float,
    max_freq: float,
) -> np.ndarray:
    usable_width = width - left_margin - right_margin
    freq_log_max = math.log(max_freq / min_freq)
    x_arr = np.asarray(x, dtype=np.float64)
    freq = min_freq * np.exp(freq_log_max * (x_arr - left_margin) / usable_width)
    return np.clip(freq, min_freq, max_freq)


def magnitude_to_01(
    magnitude: np.ndarray,
    freq: np.ndarray,
    ceiling_db: float,
    noise_floor_db: float,
    tilt_db_per_oct: float,
) -> np.ndarray:
    db = 10.0 * np.log10(np.maximum(magnitude, SMALL_MAG))
    if tilt_db_per_oct != 0.0:
        db = db + tilt_db_per_oct * (np.log2(freq) - np.log2(1024.0))

    return 1.0 - ((db - ceiling_db) / (noise_floor_db - ceiling_db))


def one_to_y(
    z01: np.ndarray,
    height: int,
    top_margin: float,
    bottom_margin: float,
    text_height: float,
) -> np.ndarray:
    usable_height = height - top_margin - bottom_margin - text_height
    return top_margin + (1.0 - z01) * usable_height


def db_to_y(
    db: float,
    height: int,
    top_margin: float,
    bottom_margin: float,
    text_height: float,
    ceiling_db: float,
    noise_floor_db: float,
) -> float:
    usable_height = height - top_margin - bottom_margin - text_height
    return top_margin + (((db - ceiling_db) / (noise_floor_db - ceiling_db)) * usable_height)


def compute_decay_delta(fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return float(0.99 ** (30.0 / fps))


def format_frequency_label(freq: int) -> str:
    return f"{int(freq / 1000)}k" if freq >= 1000 else str(freq)
