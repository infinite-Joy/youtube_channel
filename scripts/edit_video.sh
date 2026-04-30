#!/usr/bin/env bash
# edit_video.sh — full video editing pipeline
#
# Steps:
#   1. Normalize clean audio to -14 LUFS (YouTube target)
#   2. Sync normalized audio to screen recording
#   3. Cut silent + static sections from synced video
#
# Usage:
#   ./edit_video.sh <screen_recording> <clean_audio> <output>
#
# Example:
#   ./edit_video.sh "Screen_Recording.mp4" "mic_audio.wav" "final.mp4"

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <screen_recording> <clean_audio> <output>"
    echo ""
    echo "Example:"
    echo "  $0 \"Screen_Recording.mp4\" \"mic_audio.wav\" \"final.mp4\""
    exit 1
fi

VIDEO="$1"
AUDIO="$2"
OUTPUT="$3"

# Intermediate files sit next to the output
DIR="$(dirname "$OUTPUT")"
BASE="$(basename "$OUTPUT" .mp4)"
NORMALIZED_AUDIO="${DIR}/${BASE}_step1_normalized.wav"
SYNCED_VIDEO="${DIR}/${BASE}_step2_synced.mp4"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

step() { echo -e "\n${BOLD}${GREEN}▶ $*${RESET}"; }
info() { echo -e "  ${YELLOW}$*${RESET}"; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
for f in "$VIDEO" "$AUDIO"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

for cmd in python3 ffmpeg; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found on PATH"
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate the uv virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Step 1 — Normalize audio to -14 LUFS
# ---------------------------------------------------------------------------
step "Step 1/3 — Normalize audio to -14 LUFS (YouTube target)"
info "Input : $AUDIO"
info "Output: $NORMALIZED_AUDIO"

python3 "${SCRIPT_DIR}/loudness_analyze.py" \
    "$AUDIO" \
    --apply-gain \
    --output "$NORMALIZED_AUDIO"

# ---------------------------------------------------------------------------
# Step 2 — Sync normalized audio to screen recording
# ---------------------------------------------------------------------------
step "Step 2/3 — Sync audio to screen recording via MFCC cross-correlation"
info "Video : $VIDEO"
info "Audio : $NORMALIZED_AUDIO"
info "Output: $SYNCED_VIDEO"

python3 "${SCRIPT_DIR}/sync_video_audio.py" \
    "$VIDEO" \
    "$NORMALIZED_AUDIO" \
    --output "$SYNCED_VIDEO"

# ---------------------------------------------------------------------------
# Step 3 — Cut silent + static sections
# ---------------------------------------------------------------------------
step "Step 3/3 — Cut silent and static sections"
info "Input : $SYNCED_VIDEO"
info "Output: $OUTPUT"

LOG_FILE="${DIR}/${BASE}_cuts.log"

python3 "${SCRIPT_DIR}/cut_silent_static.py" \
    "$SYNCED_VIDEO" \
    --out "$OUTPUT" \
    --log "$LOG_FILE"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}✓ Pipeline complete${RESET}"
echo "  Final output : $OUTPUT"
echo "  Cuts log     : $LOG_FILE"

# Clean up intermediates
rm -f "$NORMALIZED_AUDIO" "$SYNCED_VIDEO"
info "Intermediate files removed."
