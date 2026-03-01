from __future__ import annotations

import numpy as np
import pytest

from spectrumov.renderer import (
    ReSpectrumRenderer,
    SpectrumRenderConfig,
    choose_audio_codec_for_container,
    decode_audio_mono_with_ffmpeg,
    load_audio_mono_for_analysis,
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


def test_load_audio_wav_path_uses_wavfile_branch(tmp_path) -> None:
    from scipy.io import wavfile

    cfg = SpectrumRenderConfig(ffmpeg_path="/not/used")
    wav_path = tmp_path / "input.wav"
    samples = np.array([[1000, -1000], [2000, -2000], [0, 0]], dtype=np.int16)
    wavfile.write(str(wav_path), 44100, samples)

    sample_rate, mono = load_audio_mono_for_analysis(wav_path, cfg)

    assert sample_rate == 44100
    assert mono.shape == (3,)
    assert mono.dtype == np.float32
    # L+R on symmetric samples should cancel out.
    np.testing.assert_allclose(mono, np.zeros(3, dtype=np.float32), atol=1e-6)


def test_load_audio_non_wav_uses_ffmpeg_decoder(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_decode(input_audio, ffmpeg_path, sample_rate):
        captured["input_audio"] = input_audio
        captured["ffmpeg_path"] = ffmpeg_path
        captured["sample_rate"] = sample_rate
        return np.array([0.1, -0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr("spectrumov.renderer.decode_audio_mono_with_ffmpeg", _fake_decode)

    cfg = SpectrumRenderConfig(ffmpeg_path="/usr/bin/ffmpeg", analysis_sample_rate=48000)
    input_mp3 = tmp_path / "input.mp3"
    input_mp3.write_bytes(b"fake")

    sample_rate, mono = load_audio_mono_for_analysis(input_mp3, cfg)

    assert sample_rate == 48000
    np.testing.assert_allclose(mono, np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert str(captured["input_audio"]).endswith("input.mp3")
    assert captured["ffmpeg_path"] == "/usr/bin/ffmpeg"
    assert captured["sample_rate"] == 48000


def test_decode_audio_mono_with_ffmpeg_reads_f32(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "input.ogg"
    audio_path.write_bytes(b"fake")
    expected = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    class _Proc:
        def __init__(self):
            self.returncode = 0
            self.stdout = expected.tobytes()
            self.stderr = b""

    def _fake_run(cmd, stdout, stderr, check):
        assert "-ac" in cmd and "1" in cmd
        assert "-f" in cmd and "f32le" in cmd
        assert check is False
        return _Proc()

    monkeypatch.setattr("spectrumov.renderer.subprocess.run", _fake_run)

    decoded = decode_audio_mono_with_ffmpeg(audio_path, "/usr/bin/ffmpeg", 48000)
    np.testing.assert_allclose(decoded, expected)


def test_decode_audio_mono_with_ffmpeg_raises_on_error(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "input.flac"
    audio_path.write_bytes(b"fake")

    class _Proc:
        def __init__(self):
            self.returncode = 1
            self.stdout = b""
            self.stderr = b"boom"

    def _fake_run(cmd, stdout, stderr, check):
        return _Proc()

    monkeypatch.setattr("spectrumov.renderer.subprocess.run", _fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg decode failed"):
        decode_audio_mono_with_ffmpeg(audio_path, "/usr/bin/ffmpeg", 48000)
