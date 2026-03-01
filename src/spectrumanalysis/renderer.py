from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.io import wavfile
from tqdm import tqdm

from .utils import (
    SMALL_MAG,
    compute_decay_delta,
    compute_group_map,
    db_to_y,
    format_frequency_label,
    freq_to_x,
    jsfx_window,
    magnitude_to_01,
    normalize_audio_array,
    one_to_y,
    reduce_fft_to_groups,
    sum_channels_to_mono,
)


def choose_audio_codec_for_container(output_path: str | Path) -> str:
    suffix = Path(output_path).suffix.lower()
    return "pcm_s16le" if suffix == ".mov" else "aac"


@dataclass(slots=True)
class SpectrumRenderConfig:
    width: int = 1920
    height: int = 500
    fps: float = 60.0
    fft_size: int = 8192
    window_type: int = 2
    ceiling_db: float = 0.0
    noise_floor_db: float = -90.0
    tilt_db_per_oct: float = 4.5
    min_freq: float = 10.0
    # JSFX defaults to a fixed 44.1kHz Nyquist for display mapping.
    max_freq: float = 22050.0
    display_mode: str = "fill"
    show_peaks: bool = False
    show_grid: bool = False
    encoder: str = "ffmpeg"
    ffmpeg_path: str = "/usr/bin/ffmpeg"
    ffmpeg_vcodec: str = "prores_ks"
    ffmpeg_crf: int = 12
    ffmpeg_preset: str = "slow"
    ffmpeg_pix_fmt: str = "yuv422p10le"
    ffmpeg_prores_profile: int = 3
    render_bit_depth: int = 16
    curve_smoothing_sigma: float = 1.2


class ReSpectrumRenderer:
    """Offline renderer that mirrors ReSpectrum JSFX defaults in OpenCV."""

    def __init__(self, sample_rate: float, config: SpectrumRenderConfig) -> None:
        if config.display_mode not in {"fill", "line", "none"}:
            raise ValueError("display_mode must be one of: fill, line, none")
        if config.width <= 0 or config.height <= 0:
            raise ValueError("width and height must be positive")
        if config.render_bit_depth not in {8, 16}:
            raise ValueError("render_bit_depth must be 8 or 16")

        self.sample_rate = float(sample_rate)
        self.config = config
        self.frame_dtype = np.uint16 if config.render_bit_depth == 16 else np.uint8
        self.frame_max = 65535 if config.render_bit_depth == 16 else 255

        self.left_margin = 0.0
        self.right_margin = 0.0
        self.top_margin = 10.0

        self.compact_width = config.width < 600
        self.compact_height = config.height < 320
        self.very_compact_width = config.width < 450
        self.very_compact_height = config.height < 240
        self.very_compact = self.very_compact_width or self.very_compact_height

        self.text_height = 12.0
        self.bottom_margin = -18.0 if self.very_compact else 0.0
        self.bottom = config.height - self.bottom_margin - self.text_height

        self.window = jsfx_window(config.window_type, config.fft_size).astype(np.float64)

        _, self.group_ids, self.group_freqs = compute_group_map(self.sample_rate, config.fft_size)
        self.group_count = int(self.group_freqs.shape[0])

        self.group_x = freq_to_x(
            self.group_freqs,
            config.width,
            self.left_margin,
            self.right_margin,
            config.min_freq,
            config.max_freq,
        )
        self.group_x_int = np.rint(self.group_x).astype(np.int32)

        self.smoothed_y01 = np.zeros(self.group_count, dtype=np.float64)
        self.peak_mag = np.full(self.group_count, SMALL_MAG, dtype=np.float64)
        self.decay_delta = compute_decay_delta(config.fps)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.45
        self.font_thickness = 1

        # RGB in JSFX -> BGR in OpenCV.
        self.fill_color = self._scale_color((83, 71, 46))
        self.line_color = self._scale_color((253, 215, 114))
        self.peak_color = self._scale_color((204, 204, 204))

        self.grid_hline_color = self._scale_color((77, 77, 77))
        self.grid_minor_vline_color = self._scale_color((153, 153, 153))
        self.grid_major_vline_color = self._scale_color((204, 204, 204))
        self.grid_text_color = self._scale_color((102, 102, 102))

        self._build_gradient()

    def _scale_color(self, bgr_8bit: tuple[int, int, int]) -> tuple[int, int, int]:
        if self.frame_max == 255:
            return bgr_8bit
        scale = self.frame_max / 255.0
        return tuple(int(round(c * scale)) for c in bgr_8bit)

    def _build_gradient(self) -> None:
        ht = (self.config.height - self.bottom_margin - self.text_height) + 2.0
        start = int(max(0, min(self.config.height, math.floor(ht / 2.0))))
        end = int(max(start, min(self.config.height, math.floor(ht))))

        self.gradient_start = start
        self.gradient_end = end

        if end <= start:
            self.gradient_mul = None
            return

        rows = end - start
        alpha = np.linspace(0.0, 0.55, num=rows, endpoint=False, dtype=np.float32)
        self.gradient_mul = (1.0 - alpha).reshape(rows, 1, 1)

    def analyze_frame(self, frame_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if frame_samples.shape[0] != self.config.fft_size:
            raise ValueError(
                f"Expected {self.config.fft_size} samples for analysis, got {frame_samples.shape[0]}"
            )

        windowed = frame_samples.astype(np.float64, copy=False) * self.window
        fft_bins = np.fft.rfft(windowed)
        power_bins = (np.abs(fft_bins[1:]) ** 2).astype(np.float64, copy=False)

        grouped_mag = reduce_fft_to_groups(power_bins, self.group_ids, self.group_count)
        np.maximum(self.peak_mag, grouped_mag, out=self.peak_mag)

        current_y01 = magnitude_to_01(
            grouped_mag,
            self.group_freqs,
            self.config.ceiling_db,
            self.config.noise_floor_db,
            self.config.tilt_db_per_oct,
        )

        self.smoothed_y01 *= self.decay_delta
        np.maximum(self.smoothed_y01, current_y01, out=self.smoothed_y01)

        curve_y = one_to_y(
            self.smoothed_y01,
            self.config.height,
            self.top_margin,
            self.bottom_margin,
            self.text_height,
        )

        peak_y = None
        if self.config.show_peaks:
            peak_y01 = magnitude_to_01(
                self.peak_mag,
                self.group_freqs,
                self.config.ceiling_db,
                self.config.noise_floor_db,
                self.config.tilt_db_per_oct,
            )
            peak_y = one_to_y(
                peak_y01,
                self.config.height,
                self.top_margin,
                self.bottom_margin,
                self.text_height,
            )
            peak_y = np.minimum(peak_y, self.bottom)

        return curve_y, peak_y

    def _fill_polygon_soft(self, frame: np.ndarray, poly: np.ndarray, color: tuple[int, int, int]) -> None:
        # Blend fill with a lightly blurred alpha mask to soften the top edge.
        height, width = frame.shape[:2]
        pad = 2
        x0 = int(np.clip(np.min(poly[:, 0]) - pad, 0, width - 1))
        x1 = int(np.clip(np.max(poly[:, 0]) + pad + 1, 1, width))
        y0 = int(np.clip(np.min(poly[:, 1]) - pad, 0, height - 1))
        y1 = int(np.clip(np.max(poly[:, 1]) + pad + 1, 1, height))
        if x1 <= x0 or y1 <= y0:
            return

        roi_w = x1 - x0
        roi_h = y1 - y0
        local_poly = poly - np.array([x0, y0], dtype=np.int32)

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(mask, [local_poly.astype(np.int32)], 255, lineType=cv2.LINE_8)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.6, sigmaY=0.6)

        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        fill = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        frame[y0:y1, x0:x1] = np.clip(
            roi * (1.0 - alpha) + fill * alpha, 0.0, float(self.frame_max)
        ).astype(self.frame_dtype)

    def _draw_curve(self, frame: np.ndarray, curve_y: np.ndarray) -> None:
        if self.config.display_mode == "none":
            return

        points = self._build_smoothed_points(curve_y)
        if points.shape[0] < 2:
            return

        if self.config.display_mode == "fill":
            poly = np.vstack(
                (
                    np.array([[points[0, 0], int(self.bottom)]], dtype=np.int32),
                    points,
                    np.array([[points[-1, 0], int(self.bottom)]], dtype=np.int32),
                )
            )
            poly[:, 1] = np.clip(poly[:, 1], 0, self.config.height - 1)
            self._fill_polygon_soft(frame, poly, self.fill_color)
        elif self.config.display_mode == "line":
            cv2.polylines(frame, [points], False, self.line_color, 1, cv2.LINE_AA)

    def _build_smoothed_points(self, y_values: np.ndarray) -> np.ndarray:
        x_src = np.clip(self.group_x, 0.0, float(self.config.width - 1))
        y_src = np.minimum(y_values, self.bottom)
        y_src = np.clip(y_src, 0.0, float(self.config.height - 1))
        if self.config.curve_smoothing_sigma > 0.0:
            y_src = gaussian_filter1d(y_src, sigma=self.config.curve_smoothing_sigma, mode="nearest")

        x_start = int(max(0, math.ceil(float(x_src[0]))))
        x_end = int(min(self.config.width - 1, math.floor(float(x_src[-1]))))
        if x_end <= x_start:
            return np.column_stack((np.rint(x_src).astype(np.int32), np.rint(y_src).astype(np.int32)))

        x_dense = np.arange(x_start, x_end + 1, dtype=np.float64)
        try:
            spline = PchipInterpolator(x_src, y_src, extrapolate=False)
            y_dense = spline(x_dense)
        except Exception:
            # Fallback to linear interpolation if spline construction fails.
            y_dense = np.interp(x_dense, x_src, y_src)

        y_dense = np.nan_to_num(y_dense, nan=self.bottom, posinf=self.bottom, neginf=0.0)
        y_dense = np.clip(np.rint(y_dense), 0, self.config.height - 1).astype(np.int32)
        return np.column_stack((x_dense.astype(np.int32), y_dense))

    def _draw_peaks(self, frame: np.ndarray, peak_y: np.ndarray | None) -> None:
        if not self.config.show_peaks or peak_y is None:
            return

        x = np.clip(self.group_x_int, 0, self.config.width - 1)
        y = np.minimum(peak_y, self.bottom)
        y = np.clip(np.rint(y).astype(np.int32), 0, self.config.height - 1)
        points = np.column_stack((x, y)).astype(np.int32)

        if points.shape[0] >= 2:
            cv2.polylines(frame, [points], False, self.peak_color, 1, cv2.LINE_AA)

    def _apply_gradient(self, frame: np.ndarray) -> None:
        if self.gradient_mul is None:
            return

        view = frame[self.gradient_start : self.gradient_end]
        scaled = (view.astype(np.float32) * self.gradient_mul).astype(self.frame_dtype)
        frame[self.gradient_start : self.gradient_end] = scaled

    def _draw_grid(self, frame: np.ndarray) -> None:
        if not self.config.show_grid or self.very_compact:
            return

        last_text_y = -100.0
        db = self.config.ceiling_db
        while db >= self.config.noise_floor_db - 1e-9:
            y = db_to_y(
                db,
                self.config.height,
                self.top_margin,
                self.bottom_margin,
                self.text_height,
                self.config.ceiling_db,
                self.config.noise_floor_db,
            )

            if y > last_text_y:
                yi = int(round(y))
                cv2.line(
                    frame,
                    (int(self.text_height * 2), yi),
                    (self.config.width - 1, yi),
                    self.grid_hline_color,
                    1,
                    cv2.LINE_AA,
                )

                label = str(int(db))
                text_baseline = int(round(y - (self.text_height * 0.5) + self.text_height))
                cv2.putText(
                    frame,
                    label,
                    (0, text_baseline),
                    self.font,
                    0.42,
                    self.grid_text_color,
                    1,
                    cv2.LINE_AA,
                )

                last_text_y = y - (self.text_height * 0.5) + self.text_height

            db -= 10.0

        f = 10
        lx = 0.0
        gfx_x = 0.0
        major_labels = {20, 50, 100, 200, 500, 1000, 2000, 5000, 10000}

        while f < self.config.max_freq:
            tx = float(
                freq_to_x(
                    f,
                    self.config.width,
                    self.left_margin,
                    self.right_margin,
                    self.config.min_freq,
                    self.config.max_freq,
                )
            )

            dotext = tx > gfx_x and f in major_labels

            if tx > lx:
                lx = tx + 4.0
                y_end = self.config.height - (12 if dotext else self.text_height + 2) - self.bottom_margin
                color = self.grid_major_vline_color if dotext else self.grid_minor_vline_color
                cv2.line(
                    frame,
                    (int(round(tx)), 0),
                    (int(round(tx)), int(round(y_end))),
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if dotext:
                label = format_frequency_label(f)
                (tw, _), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
                gfx_x = tx - (tw * 0.5)
                baseline_y = int(round(self.config.height - (self.text_height - 2) - self.bottom_margin))
                cv2.putText(
                    frame,
                    label,
                    (int(round(tx - tw * 0.5)), baseline_y),
                    self.font,
                    self.font_scale,
                    self.grid_text_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

            if f < 100:
                f += 10
            elif f < 1000:
                f += 100
            elif f < 10000:
                f += 1000
            else:
                f += 10000

    def render_frame(self, frame_samples: np.ndarray) -> np.ndarray:
        curve_y, peak_y = self.analyze_frame(frame_samples)

        frame = np.zeros((self.config.height, self.config.width, 3), dtype=self.frame_dtype)

        self._draw_curve(frame, curve_y)
        self._draw_peaks(frame, peak_y)
        self._apply_gradient(frame)
        self._draw_grid(frame)

        return frame

    def render_audio(
        self,
        mono_audio: np.ndarray,
        output_mp4: str | Path,
        show_progress: bool = True,
        source_audio_path: str | Path | None = None,
    ) -> int:
        output_path = Path(output_mp4)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = max(1, int(math.ceil((mono_audio.shape[0] / self.sample_rate) * self.config.fps)))
        sample_step = self.sample_rate / self.config.fps
        frame_ends = np.minimum(
            np.floor((np.arange(total_frames, dtype=np.float64) + 1.0) * sample_step).astype(np.int64),
            mono_audio.shape[0],
        )

        padded = np.pad(mono_audio.astype(np.float32, copy=False), (self.config.fft_size, 0))

        iterator = frame_ends
        if show_progress:
            iterator = tqdm(frame_ends, total=total_frames, desc="Rendering", unit="frame")

        if self.config.encoder == "opencv":
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.config.fps),
                (self.config.width, self.config.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open OpenCV video writer for: {output_path}")

            try:
                for end in iterator:
                    block = padded[end : end + self.config.fft_size]
                    frame = self.render_frame(block)
                    if frame.dtype != np.uint8:
                        frame = (frame / 257.0).astype(np.uint8)
                    writer.write(frame)
            finally:
                writer.release()
        elif self.config.encoder == "ffmpeg":
            input_pix_fmt = "bgr48le" if self.frame_dtype == np.uint16 else "bgr24"
            cmd = [
                self.config.ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                input_pix_fmt,
                "-s",
                f"{self.config.width}x{self.config.height}",
                "-r",
                str(float(self.config.fps)),
                "-i",
                "-",
            ]
            if source_audio_path is not None:
                audio_codec = choose_audio_codec_for_container(output_path)
                cmd += [
                    "-i",
                    str(source_audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:a",
                    audio_codec,
                    "-shortest",
                ]
                if audio_codec == "aac":
                    cmd += ["-b:a", "192k"]
            else:
                cmd += ["-an"]
            cmd += [
                "-c:v",
                self.config.ffmpeg_vcodec,
            ]
            if self.config.ffmpeg_vcodec in {"libx264", "libx265"}:
                cmd += [
                    "-preset",
                    self.config.ffmpeg_preset,
                    "-crf",
                    str(int(self.config.ffmpeg_crf)),
                ]
            elif self.config.ffmpeg_vcodec == "prores_ks":
                cmd += [
                    "-profile:v",
                    str(int(self.config.ffmpeg_prores_profile)),
                ]
            cmd += [
                "-pix_fmt",
                self.config.ffmpeg_pix_fmt,
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                assert proc.stdin is not None
                for end in iterator:
                    block = padded[end : end + self.config.fft_size]
                    frame = self.render_frame(block)
                    proc.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                stderr = b""
                if proc.stderr is not None:
                    stderr = proc.stderr.read()
                raise RuntimeError(f"ffmpeg pipe failed: {stderr.decode(errors='replace')}") from exc
            finally:
                if proc.stdin is not None:
                    proc.stdin.close()

            stderr_out = b""
            if proc.stderr is not None:
                stderr_out = proc.stderr.read()
                proc.stderr.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {rc}: {stderr_out.decode(errors='replace')}")
        else:
            raise ValueError(f"Unsupported encoder: {self.config.encoder}")

        return total_frames


def render_wav_to_mp4(
    input_wav: str | Path,
    output_mp4: str | Path,
    config: SpectrumRenderConfig,
    show_progress: bool = True,
) -> dict[str, int | float | str]:
    sample_rate, samples = wavfile.read(str(input_wav))
    normalized = normalize_audio_array(samples)
    mono = sum_channels_to_mono(normalized)

    renderer = ReSpectrumRenderer(float(sample_rate), config)
    frames = renderer.render_audio(
        mono,
        output_mp4,
        show_progress=show_progress,
        source_audio_path=input_wav,
    )

    return {
        "input_wav": str(input_wav),
        "output_mp4": str(output_mp4),
        "sample_rate": int(sample_rate),
        "samples": int(mono.shape[0]),
        "frames": int(frames),
        "fps": float(config.fps),
        "width": int(config.width),
        "height": int(config.height),
    }
