# spectrumov

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
Default render settings are optimized for post-production: ProRes 422 HQ (`prores_ks`, profile 3),
10-bit `yuv422p10le`, and uncompressed PCM audio (`pcm_s16le`) in `.mov`.
For explicit `.mp4` outputs, audio is muxed as AAC.

## Usage

```bash
poetry run python -m spectrumov input.wav output.mov
```

Default video size is `1920x500`. You can override it:

```bash
poetry run python -m spectrumov input.wav output.mov --width 1280 --height 360
```

Common options:

```bash
poetry run python -m spectrumov input.wav output.mov \
  --fps 30 \
  --encoder ffmpeg \
  --display-mode fill \
  --curve-smoothing 1.2 \
  --show-peaks \
  --show-grid \
  --fft-size 8192 \
  --window-type blackman-harris
```

Grid rendering is hidden by default; pass `--show-grid` to enable it.
Default encoder is `ffmpeg` (`/usr/bin/ffmpeg`) using `prores_ks`.
Internal rendering defaults to 16-bit before encoding (`--render-bit-depth 16`).
Curve smoothing defaults to `--curve-smoothing 1.2` (set `0` to disable).

If output is omitted, the output defaults to the input filename with `.mov` (PCM audio).

## Tests

```bash
poetry run python -m pytest
```
