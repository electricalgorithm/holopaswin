#!/usr/bin/env bash
# Generate the test sets of the distance sweep (scripts/z_sweep.py).
#
# One hologen dataset per distance offset, all from the same seeds and recipe, so
# that the sets differ only in the distance at which the holograms are recorded.
# The seeds must not overlap the training set (hologen's default base seed is
# 10000, 25,000 holograms), so the sweep starts at 900000.
#
# Usage: scripts/run_z_sweep_sets.sh OUTPUT_ROOT [SAMPLES] [RECIPE]
#   OUTPUT_ROOT  folder that receives one dataset per offset (offset_+4, ...)
#   SAMPLES      holograms per offset (default 400)
#   RECIPE       hologen recipe file (default: hologen's default recipe)
set -euo pipefail

OUTPUT_ROOT="${1:?usage: $0 OUTPUT_ROOT [SAMPLES] [RECIPE]}"
SAMPLES="${2:-400}"
RECIPE="${3:-}"
SEED=900000
OFFSETS=(-10 -8 -6 -4 -2 -1 0 1 2 4 6 8 10)  # um; 8 and 10 lie outside the +-6 um training range
HOLOGEN_DIR="$(cd "$(dirname "$0")/../../hologen" && pwd)"

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
for OFFSET in "${OFFSETS[@]}"; do
    NAME="$(printf 'offset_%+d' "${OFFSET}")"
    echo "=== ${NAME}"
    (cd "${HOLOGEN_DIR}" && uv run hologen generate --output "${OUTPUT_ROOT}/${NAME}" --samples "${SAMPLES}" \
        --seed "${SEED}" --z-offset "${OFFSET}" --no-visualization ${RECIPE:+--recipe "${RECIPE}"})
done
