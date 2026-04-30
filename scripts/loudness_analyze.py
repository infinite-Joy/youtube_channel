#!/usr/bin/env python3
"""
Analyze audio loudness and export a YouTube-ready normalized file (-14 LUFS / -1 dBTP).

Uses a two-pass ffmpeg loudnorm (EBU R128) export so that even audio whose true peak
is already near 0 dBTP can be brought up to -14 LUFS via transparent dynamic limiting.

Usage:
    python loudness_analyze.py "fa fwd kernel.wav"
    python loudness_analyze.py "fa fwd kernel.wav" --apply-gain
    python loudness_analyze.py "fa fwd kernel.wav" --apply-gain --output out.wav
"""

import argparse
import subprocess
import sys
import tempfile
import os
import math

import numpy as np
import soundfile as sf
import pyloudnorm as pyln


TARGET_LUFS = -14.0
PEAK_LIMIT_DBTP = -1.0
NOISE_GAIN_WARNING_DB = 20.0


def decode_to_wav(input_path: str) -> tuple[str, bool]:
    """Decode non-WAV audio to a temp WAV file. Returns (path, is_temp)."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".wav",):
        return input_path, False

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "48000", "-ac", "2",
        "-acodec", "pcm_f32le", tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp.name)
        print(f"ERROR: ffmpeg failed to decode '{input_path}':\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return tmp.name, True


def measure(audio: np.ndarray, rate: int) -> tuple[float, float, float]:
    """Return (integrated_lufs, true_peak_dbtp, lra)."""
    meter = pyln.Meter(rate)  # EBU R128

    # Integrated loudness
    loudness = meter.integrated_loudness(audio)

    # True peak via 4× oversampling (ITU-R BS.1770-4 method)
    from scipy.signal import resample_poly
    if audio.ndim == 1:
        oversampled = resample_poly(audio, 4, 1)
        peak_linear = float(np.max(np.abs(oversampled)))
    else:
        peak_linear = 0.0
        for ch in range(audio.shape[1]):
            oversampled = resample_poly(audio[:, ch], 4, 1)
            peak_linear = max(peak_linear, float(np.max(np.abs(oversampled))))

    true_peak_dbtp = 20.0 * math.log10(peak_linear) if peak_linear > 0 else -math.inf

    # Loudness range
    lra = meter.loudness_range(audio)

    return loudness, true_peak_dbtp, lra


def calculate_gain(integrated_lufs: float, true_peak_dbtp: float) -> tuple[float, bool]:
    """Return (gain_db, was_peak_clamped)."""
    gain = TARGET_LUFS - integrated_lufs
    projected_peak = true_peak_dbtp + gain
    clamped = False
    if projected_peak > PEAK_LIMIT_DBTP:
        gain = PEAK_LIMIT_DBTP - true_peak_dbtp
        clamped = True
    return gain, clamped


def loudnorm_export(input_path: str, output_path: str) -> None:
    """
    Two-pass ffmpeg loudnorm export targeting -14 LUFS / -1 dBTP.

    Pass 1 measures the file's true loudness stats; pass 2 applies the
    normalization with those stats so ffmpeg can use a linear gain + limiter
    combination for the best quality result.
    """
    import json

    filter_base = f"loudnorm=I={TARGET_LUFS}:TP={PEAK_LIMIT_DBTP}:LRA=11"

    # --- Pass 1: measure ---
    cmd1 = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"{filter_base}:print_format=json",
        "-vn", "-f", "null", "-",
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        print(f"ERROR: ffmpeg pass 1 failed:\n{result1.stderr}", file=sys.stderr)
        sys.exit(1)

    # loudnorm JSON is written to stderr by ffmpeg
    stderr = result1.stderr
    json_start = stderr.rfind("{")
    json_end = stderr.rfind("}") + 1
    if json_start == -1:
        print("ERROR: Could not parse loudnorm JSON from ffmpeg output.", file=sys.stderr)
        sys.exit(1)
    stats = json.loads(stderr[json_start:json_end])

    measured_I      = stats["input_i"]
    measured_LRA    = stats["input_lra"]
    measured_TP     = stats["input_tp"]
    measured_thresh = stats["input_thresh"]
    offset          = stats["target_offset"]

    # --- Pass 2: apply ---
    filter_p2 = (
        f"{filter_base}"
        f":measured_I={measured_I}"
        f":measured_LRA={measured_LRA}"
        f":measured_TP={measured_TP}"
        f":measured_thresh={measured_thresh}"
        f":offset={offset}"
        f":linear=true"
        f":print_format=summary"
    )
    cmd2 = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", filter_p2,
        "-acodec", "pcm_s24le",
        output_path,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"ERROR: ffmpeg pass 2 failed:\n{result2.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"\nNormalized file written to: {output_path}")


def print_report(
    input_path: str,
    integrated: float,
    true_peak: float,
    lra: float,
    gain: float,
    clamped: bool,
) -> None:
    sep = "-" * 50
    print(sep)
    print(f"File            : {os.path.basename(input_path)}")
    print(sep)
    print(f"Integrated LUFS : {integrated:+.1f} LUFS")
    print(f"True Peak       : {true_peak:+.1f} dBTP")
    print(f"Loudness Range  : {lra:.1f} LU")
    print(sep)
    print(f"Target LUFS     : {TARGET_LUFS:+.1f} LUFS")
    print(f"Peak Limit      : {PEAK_LIMIT_DBTP:+.1f} dBTP")
    print(sep)
    print(f"Recommended Gain: {gain:+.1f} dB", end="")
    if clamped:
        print("  [peak-limited — export uses loudnorm limiter to reach target]")
    else:
        print()

    # Warnings
    if gain > NOISE_GAIN_WARNING_DB:
        print(f"\nWARNING: Gain is {gain:.1f} dB. This much amplification will also")
        print("         raise the noise floor significantly.")
    if true_peak > 0.0:
        print(f"\nWARNING: True peak is already above 0 dBTP ({true_peak:+.1f} dBTP).")
        print("         Consider applying a limiter before normalizing.")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure loudness and calculate YouTube normalization gain."
    )
    parser.add_argument("input", help="Input audio file (WAV, FLAC, MP3, AAC, MP4, …)")
    parser.add_argument(
        "--apply-gain",
        action="store_true",
        help="Apply calculated gain and write a normalized WAV alongside the input.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for normalized file (default: <input>_normalized.wav)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: File not found: '{args.input}'", file=sys.stderr)
        sys.exit(1)

    # Decode if needed
    wav_path, is_temp = decode_to_wav(args.input)

    try:
        audio, rate = sf.read(wav_path, dtype="float64", always_2d=True)
    finally:
        if is_temp:
            os.unlink(wav_path)

    # Near-silence guard
    if np.max(np.abs(audio)) < 1e-9:
        print("ERROR: Audio appears to be silence or empty.", file=sys.stderr)
        sys.exit(1)

    integrated, true_peak, lra = measure(audio, rate)

    if integrated == -math.inf or not math.isfinite(integrated):
        print("ERROR: Could not measure integrated loudness (audio may be too short or silent).",
              file=sys.stderr)
        sys.exit(1)

    gain, clamped = calculate_gain(integrated, true_peak)
    print_report(args.input, integrated, true_peak, lra, gain, clamped)

    if args.apply_gain:
        if args.output:
            out_path = args.output
        else:
            base, _ = os.path.splitext(args.input)
            out_path = base + "_normalized.wav"
        loudnorm_export(args.input, out_path)


if __name__ == "__main__":
    main()
