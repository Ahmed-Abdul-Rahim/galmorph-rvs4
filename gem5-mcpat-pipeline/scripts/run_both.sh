#!/usr/bin/env bash
# Runs build/bench (RVV) and build/bench-scalar through gem5, one after
# the other. Each run writes its own stats.txt and config.json.
#
# Edit the paths in the "settings" block below once, then just run:
#   ./scripts/run_both.sh
set -euo pipefail

# ---- settings: edit these once for your machine -------------------------
GEM5_BIN="${GEM5_BIN:-$HOME/gem5/build/RISCV/gem5.opt}"
REPO_DIR="${REPO_DIR:-$HOME/galmorph-rvs4-main}"      # the repo with the patched source applied
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU="${CPU:-minor}"          # minor | timing | atomic  (see configs/se_config.py)
VLEN="${VLEN:-256}"
ELEN="${ELEN:-32}"
CLK_FREQ="${CLK_FREQ:-100MHz}"
# ---------------------------------------------------------------------------

RVV_BIN="$REPO_DIR/build/bench"
SCALAR_BIN="$REPO_DIR/build/bench-scalar"

for f in "$GEM5_BIN" "$RVV_BIN" "$SCALAR_BIN"; do
    if [ ! -f "$f" ]; then
        echo "Missing file: $f"
        echo "Build gem5 first (02_build_gem5.sh), and build both binaries in the repo:"
        echo "  make bench          # produces build/bench"
        echo "  make bench-scalar   # produces build/bench-scalar (needs the patch in patched-source/)"
        exit 1
    fi
done

echo "== Running RVV build =="
"$GEM5_BIN" --outdir "$PIPELINE_DIR/m5out-rvv" --remote-gdb-port 0 \
    "$PIPELINE_DIR/configs/se_config.py" \
    --binary "$RVV_BIN" --cpu "$CPU" --vlen "$VLEN" --elen "$ELEN" --clk-freq "$CLK_FREQ"

echo "== Running scalar build =="
"$GEM5_BIN" --outdir "$PIPELINE_DIR/m5out-scalar" --remote-gdb-port 0 \
    "$PIPELINE_DIR/configs/se_config.py" \
    --binary "$SCALAR_BIN" --cpu "$CPU" --vlen "$VLEN" --elen "$ELEN" --clk-freq "$CLK_FREQ"

echo "Done. stats.txt and config.json are in:"
echo "  $PIPELINE_DIR/m5out-rvv/"
echo "  $PIPELINE_DIR/m5out-scalar/"