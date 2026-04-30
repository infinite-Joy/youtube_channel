r"""
cut_silent_static.py â€” trim silent+static sections from a video.

Scans a video file, detects intervals where audio is silent and video shows
little or no motion, and removes those intervals.  Sections that are silent
but have motion are sped up instead.  Produces a trimmed video and prints or
writes a log of every removed/sped-up timeframe.

Designed for recorded talks, screencasts, and static-camera recordings.

Requirements:
    - Python >= 3.13
    - ffmpeg on PATH (used by MoviePy for video I/O and for silence detection)
    - Python packages: moviepy, numpy, pillow (managed via uv)

Usage:
    python cut_silent_static.py input.mp4 --out trimmed.mp4
    python cut_silent_static.py input.mp4 --out trimmed.mp4 --log removed.log
    python cut_silent_static.py input.mp4 --out trimmed.mp4 --sample_fps 2 --motion_thresh 2.0
    python cut_silent_static.py input.mp4 --min_silence_len 500 --silence_thresh -35

Parameters:
    input               Path to source video file.
    --out               Output filename for trimmed video (default: output.mp4).
    --log               Optional path to write removed timeframes log.
    --silence_thresh    Audio silence threshold in dBFS (default: -40).
                        Use -30 for noisy recordings, -50 for quiet environments.
    --min_silence_len   Minimum silence length in ms (default: 700).
    --motion_thresh     Per-frame mean absolute difference threshold for
                        motion detection (default: 2.5). Higher = more frames
                        treated as static.
    --sample_fps        Frames per second sampled for motion detection
                        (default: 1.0). Higher = finer detection, more CPU.
    --gap               Seconds of natural silence kept at each edge of a cut
                        (default: 0.35). Increase to preserve pauses around cuts.
    --speed_factor      Playback speed for silent-but-active sections
                        (default: 3.0). Use 2.0 for subtle, 4.0+ for aggressive.
    --min_speedup_len   Minimum duration in seconds before a silent-active
                        section is sped up (default: 1.0).
    --max_silent_duration  Maximum duration in seconds for a sped-up silent
                        section (default: 5.0). If a section is still longer
                        than this after speed_factor, a higher speed is used.

Behavior:
    - Intervals where BOTH silence and low motion overlap are removed entirely
      (shrunk inward by --gap to keep natural pauses at cut boundaries).
    - Intervals that are silent but have motion are sped up by --speed_factor.
    - Overlapping intervals are merged before cutting.
"""

import os
import re
import subprocess
import sys
import tempfile
import argparse
import math
import logging
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips  # moviepy v1
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips  # moviepy v2
import numpy as np
from tqdm import tqdm

def format_time(t):
    """Format seconds as HH:MM:SS.mmm"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def _get_duration(video_path):
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _has_video_stream(path):
    """Return True if the file contains at least one video stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "video" in result.stdout.lower()


def detect_silence_ranges(video_path, min_silence_len=700, silence_thresh=-40):
    """Detect silence using ffmpeg silencedetect filter."""
    duration_s = _get_duration(video_path)
    duration_ms = min_silence_len / 1000.0
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={silence_thresh}dB:d={duration_ms}",
        "-f", "null", "-"
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    pbar = tqdm(
        total=duration_s,
        desc="Detecting silence",
        unit="s",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}s",
    ) if duration_s else None
    output_lines = []
    for line in proc.stderr:
        output_lines.append(line)
        time_match = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
        if time_match and pbar:
            h, m, s = time_match.groups()
            current = int(h) * 3600 + int(m) * 60 + float(s)
            pbar.n = min(current, pbar.total)
            pbar.refresh()
    proc.wait()
    if pbar:
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()
    output = "".join(output_lines)
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", output)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", output)]
    # If last silence_start has no end, use duration
    if len(starts) > len(ends):
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", output)
        if dur_match:
            h, m, s = dur_match.groups()
            dur = int(h)*3600 + int(m)*60 + float(s)
        else:
            dur = starts[-1] + 1
        ends.append(dur)
    return list(zip(starts, ends))

def detect_low_motion_ranges(video_path, sample_fps=1, motion_thresh=2.5, min_static_len=0.7):
    clip = VideoFileClip(video_path)
    dur = clip.duration
    times = np.arange(0, dur, 1.0/sample_fps)
    prev = None
    motion_flags = []
    for t in tqdm(
        times,
        desc="Detecting motion",
        unit="frame",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} frames [{elapsed}<{remaining}]",
    ):
        frame = clip.get_frame(t)
        gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
        if prev is None:
            motion = 999.0
        else:
            motion = float(np.mean(np.abs(gray - prev)))
        motion_flags.append((t, motion))
        prev = gray
    clip.close()
    low_motion_intervals = []
    start = None
    for t, m in motion_flags:
        is_low = m < motion_thresh
        if is_low and start is None:
            start = t
        if (not is_low) and start is not None:
            if t - start >= min_static_len:
                low_motion_intervals.append((start, t))
            start = None
    if start is not None and times[-1] - start >= min_static_len:
        low_motion_intervals.append((start, times[-1]))
    return low_motion_intervals

def merge_intervals(intervals):
    """Merge overlapping/adjacent intervals"""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s,e in intervals[1:]:
        last = merged[-1]
        if s <= last[1] + 1e-6:
            last[1] = max(last[1], e)
        else:
            merged.append([s,e])
    return [(a,b) for a,b in merged]

def subtract_intervals(base, subtract):
    """Subtract a list of intervals from base intervals, returning the remainder."""
    result = []
    for bs, be in base:
        remaining = [(bs, be)]
        for ss, se in subtract:
            next_remaining = []
            for rs, re in remaining:
                if se <= rs or ss >= re:
                    next_remaining.append((rs, re))
                else:
                    if rs < ss:
                        next_remaining.append((rs, ss))
                    if se < re:
                        next_remaining.append((se, re))
            remaining = next_remaining
        result.extend(remaining)
    return result

def speedx_clip(clip, factor):
    """Speed up a clip by factor, compatible with moviepy v1 and v2."""
    try:
        from moviepy import vfx as mpy_vfx
        return clip.with_effects([mpy_vfx.MultiplySpeed(factor)])
    except (ImportError, AttributeError):
        pass
    try:
        from moviepy.editor import vfx as mpy_vfx
        return clip.fx(mpy_vfx.speedx, factor)
    except (ImportError, AttributeError):
        pass
    return clip.with_speed_scaled(factor)

def merge_and_cut(video_path, silent_ranges, static_ranges, gap=0.35, speed_factor=3.0,
                  min_speedup_len=1.0, max_silent_duration=5.0, out="output.mp4",
                  log_path=None):
    clip = VideoFileClip(video_path)
    duration = clip.duration

    # REMOVE: silent AND static, shrunk inward by gap so natural silence
    # is kept at each edge of every cut.
    remove = []
    for s1, e1 in silent_ranges:
        for s2, e2 in static_ranges:
            s = max(s1, s2) + gap   # start removal *after* gap of silence
            e = min(e1, e2) - gap   # end removal *before* gap of silence
            if e - s > 0.3:         # only bother if meaningful content remains
                remove.append((max(0.0, s), min(duration, e)))
    remove = merge_intervals(remove)

    # SPEEDUP: silent but active (motion present) â€” subtract static and remove regions,
    # then keep only stretches long enough to be worth speeding up.
    silent_active = subtract_intervals(silent_ranges, static_ranges)
    silent_active = subtract_intervals(silent_active, remove)
    silent_active = [(s, e) for s, e in silent_active if e - s >= min_speedup_len]
    silent_active = merge_intervals(silent_active)

    # Logging setup
    logger = logging.getLogger("cut_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    if log_path:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)

    total_removed = 0.0
    total_sped_up = 0.0

    if remove:
        logger.info("Removed timeframes (silent + static):")
        for s, e in remove:
            logger.info(f"  - {format_time(s)}  -->  {format_time(e)}")
            total_removed += (e - s)
    else:
        logger.info("No silent+static intervals found. Nothing removed.")

    if silent_active:
        logger.info(f"\nSped-up timeframes (silent but active, {speed_factor}x, "
                    f"max {max_silent_duration}s):")
        for s, e in silent_active:
            seg_duration = e - s
            effective_speed = speed_factor
            if seg_duration / effective_speed > max_silent_duration:
                effective_speed = seg_duration / max_silent_duration
            saved = seg_duration * (1 - 1.0 / effective_speed)
            if effective_speed != speed_factor:
                logger.info(f"  - {format_time(s)}  -->  {format_time(e)}  "
                            f"({effective_speed:.1f}x, saves {saved:.1f}s)")
            else:
                logger.info(f"  - {format_time(s)}  -->  {format_time(e)}  (saves {saved:.1f}s)")
            total_sped_up += seg_duration
    else:
        logger.info("No silent+active intervals found. Nothing sped up.")

    # Build final clip: process events (remove/speedup) in time order;
    # everything between events is kept at normal speed.
    events = sorted(
        [(s, e, 'remove') for s, e in remove] +
        [(s, e, 'speedup') for s, e in silent_active],
        key=lambda x: x[0]
    )

    def safe_subclip(start, end):
        return clip.subclipped(max(0, start), min(end, duration - 0.001))

    segments = []
    cur = 0.0
    for s, e, kind in events:
        if s > cur + 1e-6:
            segments.append(safe_subclip(cur, s))
        if kind == 'speedup':
            seg_duration = e - s
            effective_speed = speed_factor
            if seg_duration / effective_speed > max_silent_duration:
                effective_speed = seg_duration / max_silent_duration
            segments.append(speedx_clip(safe_subclip(s, e), effective_speed))
        cur = max(cur, e)
    if cur < duration - 1e-6:
        segments.append(safe_subclip(cur, duration))

    final = concatenate_videoclips(segments) if segments else clip.subclipped(0, min(0.001, duration))
    final.write_videofile(out, codec="libx264", audio_codec="aac")
    final.close()
    clip.close()

    logger.info(f"\nTotal removed duration:  {total_removed:.3f} seconds")
    logger.info(f"Total sped-up duration:  {total_sped_up:.3f} seconds "
                f"(saves ~{total_sped_up*(1-1.0/speed_factor):.1f}s)")
    if log_path:
        logger.info(f"Log written to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="output.mp4")
    parser.add_argument("--log", default=None, help="Optional path to write removed timeframes log")
    parser.add_argument("--silence_thresh", type=int, default=-40)
    parser.add_argument("--min_silence_len", type=int, default=700)
    parser.add_argument("--motion_thresh", type=float, default=2.5)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--gap", type=float, default=0.35,
                        help="Seconds of silence kept at each edge of a cut (default: 0.35). "
                             "Higher values leave more natural pause around cuts.")
    parser.add_argument("--speed_factor", type=float, default=3.0,
                        help="Playback speed for silent-but-active sections (default: 3.0). "
                             "Use 2.0 for subtle, 4.0+ for aggressive.")
    parser.add_argument("--min_speedup_len", type=float, default=1.0,
                        help="Minimum duration in seconds before a silent-active section is "
                             "sped up (default: 1.0).")
    parser.add_argument("--max_silent_duration", type=float, default=5.0,
                        help="Maximum duration in seconds for a sped-up silent section "
                             "(default: 5.0). Sections longer than this after speed_factor "
                             "will use a higher speed to fit within this limit.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: File not found: '{args.input}'", file=sys.stderr)
        sys.exit(1)
    if not _has_video_stream(args.input):
        print(f"ERROR: '{args.input}' has no video stream. This script requires a video file, "
              f"not an audio-only file.", file=sys.stderr)
        sys.exit(1)

    silent = detect_silence_ranges(args.input, args.min_silence_len, args.silence_thresh)
    static = detect_low_motion_ranges(args.input, args.sample_fps, args.motion_thresh)
    merge_and_cut(args.input, silent, static, args.gap, args.speed_factor,
                  args.min_speedup_len, args.max_silent_duration, args.out, args.log)