#!/usr/bin/env python3
"""
Sync a screen recording with a clean external audio recording.

The screen recording's built-in mic audio acts as a fingerprint to align
against the clean external recording.

Offset detection uses a three-stage pipeline:
  1. Coarse  — per-coefficient MFCC cross-correlation over the FULL audio
               (no 60s trim; more unique content = fewer false peaks;
               13 independent correlations summed = far less ambiguity)
  2. Fine    — raw waveform cross-correlation in a ±1s window around the
               coarse estimate, giving sub-millisecond precision
  3. Verify  — the same waveform refinement repeated at t=2min to confirm
               consistency; falls back to the verification estimate if its
               confidence is higher

Usage:
    # Dry run — print offset and confidence only
    python sync_video_audio.py screen.mp4 clean_audio.wav

    # Produce synced output
    python sync_video_audio.py screen.mp4 clean_audio.wav --output synced.mp4

    # Enable drift correction (recommended for recordings > 30 min)
    python sync_video_audio.py screen.mp4 clean_audio.wav --output synced.mp4 --drift

    # Legacy methods (for comparison)
    python sync_video_audio.py screen.mp4 clean_audio.wav --method mfcc
    python sync_video_audio.py screen.mp4 clean_audio.wav --method waveform
"""

import argparse
import subprocess
import sys
import tempfile
import os

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYSIS_SR = 8000           # Hz — all speech energy lives below 4 kHz
CONFIDENCE_THRESHOLD = 5.0   # z-score minimum to trust an offset
BANDPASS_LOW = 300            # Hz
BANDPASS_HIGH = 3000          # Hz
DRIFT_SEGMENT_DURATION = 30   # seconds per drift window
DRIFT_SEGMENT_INTERVAL = 300  # seconds between drift windows


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_path: str, sample_rate: int = ANALYSIS_SR) -> None:
    """Extract mono audio from video at the given sample rate."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            output_path,
        ],
        check=True, capture_output=True,
    )


def bandpass_filter(y: np.ndarray, sr: int) -> np.ndarray:
    """300–3000 Hz bandpass: isolate speech formants, reject noise."""
    sos = butter(5, [BANDPASS_LOW, BANDPASS_HIGH], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def _normalize(y: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance."""
    y = y - np.mean(y)
    std = np.std(y)
    return y / std if std > 0 else y


def _load_full(path: str, sr: int) -> np.ndarray:
    """Load audio at analysis rate, bandpass filter, normalize."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = bandpass_filter(y, sr)
    return _normalize(y)


# ---------------------------------------------------------------------------
# Stage 1: coarse offset via per-coefficient MFCC cross-correlation
#
# Why per-coefficient instead of summing features first:
#   The old approach collapsed 13 MFCC dimensions into one 1D signal before
#   correlating. A spurious match in one dimension was enough to create a
#   false peak. Cross-correlating each dimension independently and then
#   summing the *normalized* correlations requires a genuine match across
#   all 13 dimensions simultaneously — dramatically reducing false peaks.
# ---------------------------------------------------------------------------

def _mfcc_coarse_offset(
    y_vid: np.ndarray,
    y_cln: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    hop_length: int = 128,
) -> tuple[float, float]:
    """
    Per-coefficient MFCC cross-correlation over the full audio arrays.
    Returns (offset_seconds, confidence_z_score).
    16 ms precision at 8 kHz with hop_length=128.
    """
    mfcc_vid = librosa.feature.mfcc(y=y_vid, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_cln = librosa.feature.mfcc(y=y_cln, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    n_v, n_c = mfcc_vid.shape[1], mfcc_cln.shape[1]
    corr_sum = np.zeros(n_v + n_c - 1)

    for i in range(n_mfcc):
        v = _normalize(mfcc_vid[i])
        c = _normalize(mfcc_cln[i])
        corr = signal.correlate(v, c, mode="full", method="fft")
        std = np.std(corr)
        if std > 0:
            corr_sum += corr / std   # normalize before summing → equal weight per coeff

    lags = signal.correlation_lags(n_v, n_c, mode="full")
    peak_idx = int(np.argmax(corr_sum))
    offset = int(lags[peak_idx]) * hop_length / sr

    std = np.std(corr_sum)
    conf = float((corr_sum[peak_idx] - np.mean(corr_sum)) / std) if std > 0 else 0.0

    return offset, conf


# ---------------------------------------------------------------------------
# Stage 2: fine offset via waveform cross-correlation in a narrow window
#
# Strategy: take a 30s chunk starting at `chunk_start_s` from the video
# audio. Search for that exact chunk in the clean audio within
# ±search_radius_s of where the coarse offset predicts it should be.
# This is O(N) rather than O(N log N) for the full audio, and gives
# sample-level (~0.1 ms) precision.
# ---------------------------------------------------------------------------

def _waveform_refine(
    y_vid: np.ndarray,
    y_cln: np.ndarray,
    sr: int,
    coarse_offset: float,
    chunk_start_s: float = 5.0,
    chunk_len_s: float = 30.0,
    search_radius_s: float = 1.0,
) -> tuple[float, float]:
    """
    Refine `coarse_offset` to sample precision.

    offset = (position of vid_chunk in y_cln) - (position of vid_chunk in y_vid)

    Returns (refined_offset_seconds, confidence_z_score).
    Returns (coarse_offset, 0.0) if there is not enough audio to search.
    """
    chunk_start = int(chunk_start_s * sr)
    chunk_end   = min(chunk_start + int(chunk_len_s * sr), len(y_vid))
    if chunk_end - chunk_start < sr:
        return coarse_offset, 0.0

    y_vid_chunk = _normalize(y_vid[chunk_start:chunk_end])

    # Expected location of vid_chunk in y_cln:
    #   y_cln[coarse_offset*sr + chunk_start]  ≈  y_vid[chunk_start]
    search_samples    = int(search_radius_s * sr)
    expected_cln_pos  = int(coarse_offset * sr) + chunk_start
    cln_start = max(0, expected_cln_pos - search_samples)
    cln_end   = min(len(y_cln), expected_cln_pos + len(y_vid_chunk) + search_samples)

    if cln_end - cln_start < len(y_vid_chunk):
        return coarse_offset, 0.0

    y_cln_window = _normalize(y_cln[cln_start:cln_end])

    # mode='valid': corr[p] = dot(y_cln_window[p : p+N], y_vid_chunk)
    # peak at p → best match starts at y_cln_window[p] = y_cln[cln_start + p]
    corr    = signal.correlate(y_cln_window, y_vid_chunk, mode="valid", method="fft")
    peak_p  = int(np.argmax(corr))

    # offset: clean position of vid_chunk start  minus  video position of chunk start
    fine_offset = (cln_start + peak_p - chunk_start) / sr

    std  = np.std(corr)
    conf = float((corr[peak_p] - np.mean(corr)) / std) if std > 0 else 0.0

    return fine_offset, conf


# ---------------------------------------------------------------------------
# Stage 3: verify at an independent window
# ---------------------------------------------------------------------------

def _verify_offset(
    y_vid: np.ndarray,
    y_cln: np.ndarray,
    sr: int,
    offset: float,
    verify_start_s: float = 120.0,
) -> tuple[float, float]:
    """Run waveform refinement at t=verify_start_s to confirm the offset."""
    return _waveform_refine(
        y_vid, y_cln, sr, offset,
        chunk_start_s=verify_start_s,
        chunk_len_s=30.0,
        search_radius_s=1.0,
    )


# ---------------------------------------------------------------------------
# Main offset entry point
# ---------------------------------------------------------------------------

def _mfcc_cosine_similarity(
    y_v: np.ndarray,
    y_c: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    hop_length: int = 128,
) -> float:
    """
    Cosine similarity between MFCC feature matrices of two equal-duration windows.
    Returns a value in [-1, 1]; >0.3 indicates good alignment.
    """
    m_v = librosa.feature.mfcc(y=y_v, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    m_c = librosa.feature.mfcc(y=y_c, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    min_f = min(m_v.shape[1], m_c.shape[1])
    fv = _normalize(m_v[:, :min_f].flatten())
    fc = _normalize(m_c[:, :min_f].flatten())
    denom = np.linalg.norm(fv) * np.linalg.norm(fc)
    return float(np.dot(fv, fc) / denom) if denom > 0 else 0.0


def _verify_alignment(
    y_vid: np.ndarray,
    y_cln: np.ndarray,
    sr: int,
    offset: float,
    check_start_s: float = 120.0,
    check_len_s: float = 30.0,
) -> float:
    """
    Confirm alignment quality at an independent window using MFCC cosine similarity.
    Does NOT search for a new offset — just scores whether the given offset is right.
    Returns cosine similarity score: >0.3 = good, <0.1 = suspect.
    """
    v_start = int(check_start_s * sr)
    v_end   = min(v_start + int(check_len_s * sr), len(y_vid))
    c_start = int(offset * sr) + v_start
    c_end   = c_start + (v_end - v_start)
    if v_end - v_start < sr or c_start < 0 or c_end > len(y_cln):
        return 0.0
    return _mfcc_cosine_similarity(
        _normalize(y_vid[v_start:v_end]),
        _normalize(y_cln[c_start:c_end]),
        sr,
    )


def find_offset_robust(
    y_vid: np.ndarray,
    y_cln: np.ndarray,
    sr: int,
) -> tuple[float, float]:
    """
    Three-stage offset detection.
    Returns (offset_seconds, confidence_z_score).

    offset > 0 → clean audio started before video → trim that many seconds
                 from the start of the clean audio before muxing.
    offset < 0 → clean audio started after video  → delay clean audio by
                 abs(offset) seconds.

    Stage 2 (waveform refinement) falls back gracefully to the Stage 1
    coarse result when the two recordings have different mic characteristics
    that make raw waveform cross-correlation unreliable.
    Stage 3 verification uses MFCC (not raw waveform) for the same reason.
    """
    print("  Stage 1: per-coefficient MFCC correlation (full audio)…")
    coarse, coarse_conf = _mfcc_coarse_offset(y_vid, y_cln, sr)
    print(f"           coarse offset = {coarse:+.4f}s  (z={coarse_conf:.1f})")

    print("  Stage 2: waveform refinement ±1s around coarse estimate…")
    fine, fine_conf = _waveform_refine(y_vid, y_cln, sr, coarse)
    print(f"           fine offset   = {fine:+.4f}s  (z={fine_conf:.1f})")

    # When waveform refinement is unreliable (different mic characteristics),
    # fall back to the MFCC coarse estimate which is robust to mic differences.
    if fine_conf < CONFIDENCE_THRESHOLD:
        print(f"           waveform confidence too low — using MFCC coarse offset")
        offset, confidence = coarse, coarse_conf
    else:
        offset, confidence = fine, fine_conf

    print("  Stage 3: alignment check at t=2 min (MFCC cosine similarity)…")
    score_early = _verify_alignment(y_vid, y_cln, sr, offset, check_start_s=120.0)
    score_late  = _verify_alignment(y_vid, y_cln, sr, offset, check_start_s=360.0)
    print(f"           t=2min score={score_early:.3f}  t=6min score={score_late:.3f}  "
          f"(>0.3 = good alignment)")

    if score_early < 0.05 and score_late < 0.05:
        print("           WARNING: both verification windows scored poorly — "
              "check that both files contain overlapping speech")

    return offset, confidence


# ---------------------------------------------------------------------------
# Legacy methods (kept for --method flag comparison)
# ---------------------------------------------------------------------------

def find_offset_mfcc(
    y_video: np.ndarray,
    y_clean: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    hop_length: int = 128,
) -> tuple[float, float]:
    """Original summed-MFCC correlation (60s window, kept for comparison)."""
    mfcc_vid = librosa.feature.mfcc(y=y_video, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_cln = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    for m in (mfcc_vid, mfcc_cln):
        std = m.std(axis=1, keepdims=True); std[std == 0] = 1.0
        m -= m.mean(axis=1, keepdims=True); m /= std
    feat_vid = mfcc_vid.sum(axis=0)
    feat_cln = mfcc_cln.sum(axis=0)
    corr = signal.correlate(feat_vid, feat_cln, mode="full", method="fft")
    lags = signal.correlation_lags(len(feat_vid), len(feat_cln), mode="full")
    peak_idx = int(np.argmax(corr))
    offset = int(lags[peak_idx]) * hop_length / sr
    std = np.std(corr)
    conf = float((corr[peak_idx] - np.mean(corr)) / std) if std > 0 else 0.0
    return offset, conf


def find_offset_waveform(
    y_video: np.ndarray,
    y_clean: np.ndarray,
    sr: int,
) -> tuple[float, float]:
    """Raw waveform FFT cross-correlation (kept for comparison)."""
    corr = signal.correlate(y_video, y_clean, mode="full", method="fft")
    lags = signal.correlation_lags(len(y_video), len(y_clean), mode="full")
    peak_idx = int(np.argmax(np.abs(corr)))
    offset = float(lags[peak_idx]) / sr
    std = np.std(corr)
    conf = float((corr[peak_idx] - np.mean(corr)) / std) if std > 0 else 0.0
    return offset, conf


# ---------------------------------------------------------------------------
# Drift detection and correction
# ---------------------------------------------------------------------------

def detect_drift(
    video_audio_path: str,
    clean_audio_path: str,
    sr: int = ANALYSIS_SR,
) -> tuple[float, float]:
    """Fit a linear model to per-segment offsets → (initial_offset, drift_rate)."""
    y_vid_full, _ = librosa.load(video_audio_path, sr=sr, mono=True)
    y_cln_full, _ = librosa.load(clean_audio_path, sr=sr, mono=True)
    total_duration = min(len(y_vid_full), len(y_cln_full)) / sr
    offsets, times = [], []
    for t in np.arange(0, total_duration - DRIFT_SEGMENT_DURATION, DRIFT_SEGMENT_INTERVAL):
        start, end = int(t * sr), int((t + DRIFT_SEGMENT_DURATION) * sr)
        y_v, y_c = _normalize(y_vid_full[start:end]), _normalize(y_cln_full[start:end])
        off, conf = find_offset_waveform(y_v, y_c, sr)
        if conf > 8.0:
            offsets.append(off); times.append(t + DRIFT_SEGMENT_DURATION / 2)
    if len(offsets) < 2:
        print("WARNING: not enough drift measurements; skipping drift correction.")
        return (offsets[0] if offsets else 0.0), 0.0
    coeffs = np.polyfit(times, offsets, 1)
    return float(coeffs[1]), float(coeffs[0])


def correct_drift(audio: np.ndarray, sr: int, drift_rate: float) -> np.ndarray:
    speed_ratio = 1.0 + drift_rate
    return librosa.resample(audio, orig_sr=int(sr * speed_ratio), target_sr=sr)


# ---------------------------------------------------------------------------
# Mux video + aligned audio
# ---------------------------------------------------------------------------

def combine_video_audio(
    video_path: str,
    clean_audio_path: str,
    output_path: str,
    offset_seconds: float,
) -> None:
    """
    Replace video audio with the aligned clean audio, copying video bit-for-bit.

    offset > 0 → trim `offset` seconds from start of clean audio  (-ss after -i)
    offset < 0 → delay clean audio by abs(offset) seconds          (-itsoffset)
    """
    cmd = ["ffmpeg", "-y", "-i", video_path]

    if offset_seconds > 0:
        # -ss before -i: input seek on the audio file (accurate for WAV — no keyframes)
        cmd += ["-ss", f"{offset_seconds:.6f}", "-i", clean_audio_path]
    elif offset_seconds < 0:
        cmd += ["-itsoffset", f"{abs(offset_seconds):.6f}", "-i", clean_audio_path]
    else:
        cmd += ["-i", clean_audio_path]

    cmd += [
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def sync_video_with_clean_audio(
    video_path: str,
    clean_audio_path: str,
    output_path: str | None,
    method: str = "robust",
    detect_drift_flag: bool = False,
    analysis_sr: int = ANALYSIS_SR,
    trim: float = 0,          # 0 = use full audio (recommended)
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        video_audio_path = os.path.join(tmpdir, "video_audio.wav")

        print("Extracting audio from video…")
        extract_audio(video_path, video_audio_path, sample_rate=analysis_sr)

        print("Loading audio…")
        y_vid = _load_full(video_audio_path, analysis_sr)
        y_cln = _load_full(clean_audio_path, analysis_sr)

        # Legacy methods accept a trimmed window
        if method in ("mfcc", "waveform") and trim > 0:
            y_vid_t = y_vid[:int(trim * analysis_sr)]
            y_cln_t = y_cln[:int(trim * analysis_sr)]
        else:
            y_vid_t, y_cln_t = y_vid, y_cln

        print(f"Computing offset via '{method}' method…")
        if method == "robust":
            offset, confidence = find_offset_robust(y_vid, y_cln, analysis_sr)
        elif method == "mfcc":
            offset, confidence = find_offset_mfcc(y_vid_t, y_cln_t, analysis_sr)
        else:
            offset, confidence = find_offset_waveform(y_vid_t, y_cln_t, analysis_sr)

        _print_offset_report(offset, confidence)

        if confidence < CONFIDENCE_THRESHOLD:
            raise ValueError(
                f"Low confidence ({confidence:.1f} < {CONFIDENCE_THRESHOLD}). "
                "Check that both files contain overlapping speech."
            )

        if output_path is None:
            print("No --output specified. Dry run complete.")
            return

        audio_to_mux = clean_audio_path
        if detect_drift_flag:
            print("Detecting clock drift…")
            initial_offset, drift_rate = detect_drift(video_audio_path, clean_audio_path, analysis_sr)
            print(f"  Drift rate  : {drift_rate*1e6:.1f} ppm  ({drift_rate*3600*1000:.1f} ms/hour)")
            print(f"  Init offset : {initial_offset:.4f} s")
            if abs(drift_rate) > 1e-6:
                print("  Applying drift correction…")
                y_full, full_sr = librosa.load(clean_audio_path, sr=None, mono=False)
                if y_full.ndim == 1:
                    y_corr = correct_drift(y_full, full_sr, drift_rate)
                else:
                    y_corr = np.stack(
                        [correct_drift(y_full[ch], full_sr, drift_rate) for ch in range(y_full.shape[0])]
                    )
                drift_path = os.path.join(tmpdir, "drift_corrected.wav")
                sf.write(drift_path, y_corr.T if y_corr.ndim > 1 else y_corr, full_sr)
                audio_to_mux = drift_path
                offset = initial_offset
            else:
                print("  Drift negligible — skipping.")

        print(f"Muxing → {output_path}")
        combine_video_audio(video_path, audio_to_mux, output_path, offset)
        print(f"\nDone. Output: {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_offset_report(offset: float, confidence: float) -> None:
    sep = "-" * 52
    direction = "trim clean audio start" if offset > 0 else "delay clean audio"
    print(sep)
    print(f"Offset      : {offset:+.4f} s  ({direction})")
    print(f"Confidence  : {confidence:.1f} z-score", end="")
    if   confidence >= 10: print("  [HIGH]")
    elif confidence >= 5:  print("  [OK]")
    else:                  print("  [LOW — result may be unreliable]")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync a screen recording with a clean external audio recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video",       help="Screen recording (MP4, MOV, MKV, …)")
    parser.add_argument("clean_audio", help="Clean external audio (WAV, FLAC, MP3, …)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (omit for dry run)")
    parser.add_argument("--method", choices=["robust", "mfcc", "waveform"], default="robust",
                        help="Offset detection method (default: robust)")
    parser.add_argument("--drift", action="store_true",
                        help="Detect and correct clock drift (for recordings > 30 min)")
    parser.add_argument("--trim", type=float, default=0,
                        help="Seconds to trim for legacy mfcc/waveform methods (0 = full audio)")
    parser.add_argument("--sr", type=int, default=ANALYSIS_SR,
                        help=f"Analysis sample rate in Hz (default: {ANALYSIS_SR})")
    args = parser.parse_args()

    for path, label in [(args.video, "video"), (args.clean_audio, "clean audio")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} file not found: '{path}'", file=sys.stderr)
            sys.exit(1)

    try:
        sync_video_with_clean_audio(
            video_path=args.video,
            clean_audio_path=args.clean_audio,
            output_path=args.output,
            method=args.method,
            detect_drift_flag=args.drift,
            analysis_sr=args.sr,
            trim=args.trim,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"ERROR: ffmpeg failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
