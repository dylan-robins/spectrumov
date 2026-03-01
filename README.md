# Spectrum Analysis

Render a ReSpectrum-style spectrum analyzer MP4 from a WAV file.

The renderer is tuned to match the default ReSpectrum JSFX behavior in this repo:

- Fill display mode
- 0 dB ceiling / -90 dB floor
- 4.5 dB/oct tilt
- Blackman-Harris window
- FFT size 8192
- JSFX-like log-frequency binning and envelope smoothing

Channel handling is mono by summing channels (`L + R`) before analysis.
When using the default ffmpeg encoder, the source WAV audio is muxed into the final video.
By default (output `.mov`), audio is muxed as uncompressed PCM (`pcm_s16le`).
For explicit `.mp4` outputs, audio is muxed as AAC.

## Usage

```bash
poetry run python -m spectrumanalysis input.wav output.mp4
```

Default video size is `1920x300`. You can override it:

```bash
poetry run python -m spectrumanalysis input.wav output.mp4 --width 1280 --height 360
```

Common options:

```bash
poetry run python -m spectrumanalysis input.wav output.mp4 \
  --fps 30 \
  --encoder ffmpeg \
  --display-mode fill \
  --show-peaks \
  --show-grid \
  --fft-size 8192 \
  --window-type blackman-harris
```

Grid rendering is hidden by default; pass `--show-grid` to enable it.
Default encoder is `ffmpeg` (`/usr/bin/ffmpeg`) using `libx264`, `-crf 12`, and `yuv444p`.

If output is omitted, the output defaults to the input filename with `.mov` (PCM audio).

## Tests

```bash
poetry run python -m pytest
```
