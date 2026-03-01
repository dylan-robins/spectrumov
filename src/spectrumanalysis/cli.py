from __future__ import annotations

import argparse
from pathlib import Path

from .renderer import SpectrumRenderConfig, render_wav_to_mp4

WINDOW_TYPES = {
    "hamming": 1,
    "blackman-harris": 2,
    "blackman": 3,
    "rectangular": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a ReSpectrum-like spectrum analyzer video from a WAV file "
            "using OpenCV."
        )
    )
    parser.add_argument("input_wav", type=Path, help="Path to source WAV file")
    parser.add_argument(
        "output_mp4",
        type=Path,
        nargs="?",
        help="Path to output video (default: <input_basename>.mov)",
    )

    parser.add_argument("--width", type=int, default=1920, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=300, help="Video height in pixels")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate")
    parser.add_argument(
        "--encoder",
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="Video encoder backend",
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default="/usr/bin/ffmpeg",
        help="Path to ffmpeg executable",
    )
    parser.add_argument(
        "--ffmpeg-crf",
        type=int,
        default=12,
        help="ffmpeg/libx264 CRF quality (lower is higher quality)",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        type=str,
        default="slow",
        help="ffmpeg/libx264 preset",
    )
    parser.add_argument(
        "--ffmpeg-pix-fmt",
        type=str,
        default="yuv444p",
        help="ffmpeg output pixel format",
    )

    parser.add_argument(
        "--display-mode",
        choices=["fill", "line", "none"],
        default="fill",
        help="Spectrum draw style",
    )
    parser.add_argument(
        "--show-peaks",
        action="store_true",
        help="Draw the optional peak overlay line",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Render frequency/dB grid lines and labels",
    )

    parser.add_argument(
        "--fft-size",
        type=int,
        default=8192,
        help="FFT size (ReSpectrum default is 8192)",
    )
    parser.add_argument(
        "--window-type",
        choices=list(WINDOW_TYPES.keys()),
        default="blackman-harris",
        help="FFT window type",
    )

    parser.add_argument("--ceiling-db", type=float, default=0.0, help="Top dB range")
    parser.add_argument("--floor-db", type=float, default=-90.0, help="Bottom dB range")
    parser.add_argument(
        "--tilt-db-oct",
        type=float,
        default=4.5,
        help="Tilt in dB/octave around 1kHz",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=10.0,
        help="Minimum displayed frequency (Hz)",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=22050.0,
        help="Maximum displayed frequency (Hz)",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_wav = args.input_wav
    if not input_wav.exists():
        raise FileNotFoundError(f"Input WAV does not exist: {input_wav}")

    output_mp4 = args.output_mp4 or input_wav.with_suffix(".mov")

    config = SpectrumRenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        fft_size=args.fft_size,
        window_type=WINDOW_TYPES[args.window_type],
        ceiling_db=args.ceiling_db,
        noise_floor_db=args.floor_db,
        tilt_db_per_oct=args.tilt_db_oct,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        display_mode=args.display_mode,
        show_peaks=args.show_peaks,
        show_grid=args.show_grid,
        encoder=args.encoder,
        ffmpeg_path=args.ffmpeg_path,
        ffmpeg_crf=args.ffmpeg_crf,
        ffmpeg_preset=args.ffmpeg_preset,
        ffmpeg_pix_fmt=args.ffmpeg_pix_fmt,
    )

    result = render_wav_to_mp4(
        input_wav=input_wav,
        output_mp4=output_mp4,
        config=config,
        show_progress=not args.no_progress,
    )

    print(
        f"Rendered {result['frames']} frames at {result['fps']} fps "
        f"to {result['output_mp4']} ({result['width']}x{result['height']})."
    )
