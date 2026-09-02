#!/usr/bin/env bash
# Put the trained checkpoints into checkpoints/. They are kept OUT of git.
# Option A: set RELEASE_URL to a base URL (e.g. a GitHub Release) that hosts the
#           files below, and this script downloads them.
# Option B: copy the files into checkpoints/ by hand (names must match).
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints
NEED=(
  "linear_d108_seed2_best.zip"
  "d64_seq256__main__seed30485.pt"
  "d108_seq256__main__seed30485.pt"
  "d64_seq4096__main__seed30485.pt"
  "d108_seq4096__main__seed30485.pt"
  "cnn_only_large_61k__recipe_main__seed_8842__noise_floor_check_second_seed.pt.zip"
)
missing=0
for f in "${NEED[@]}"; do
  if [ -f "checkpoints/$f" ]; then echo "  have  $f"; continue; fi
  if [ -n "$RELEASE_URL" ]; then
    echo "  fetch $f"; curl -fL "$RELEASE_URL/$f" -o "checkpoints/$f" || { echo "  FAILED $f"; missing=1; }
  else
    echo "  MISSING $f  (set RELEASE_URL, or copy it into checkpoints/)"; missing=1
  fi
done
[ "$missing" = 0 ] && echo "All checkpoints present." || echo "Some checkpoints missing (see above)."
