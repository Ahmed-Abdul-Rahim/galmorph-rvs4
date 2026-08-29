#!/usr/bin/env bash
# Runs build/bench-scalar on the real U74 core with a small memory system
# instead of the real board's 16GB memory reservation. See
# configs/se_config_u74_lowmem.py's docstring for exactly what this does
# and doesn't preserve from the real HiFive Unmatched board.
#
# Usage: ./scripts/run_u74_lowmem.sh
set -euo pipefail

GEM5_BIN="${GEM5_BIN:-$HOME/gem5/build/RISCV/gem5.opt}"
REPO_DIR="${REPO_DIR:-$HOME/galmorph-rvs4-main}"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLK_FREQ="${CLK_FREQ:-1.2GHz}"

SCALAR_BIN="$REPO_DIR/build/bench-scalar"

for f in "$GEM5_BIN" "$SCALAR_BIN"; do
    if [ ! -f "$f" ]; then
        echo "Missing file: $f"
        exit 1
    fi
done

echo "== Running scalar build on U74 (real core, small memory) =="
"$GEM5_BIN" --outdir "$PIPELINE_DIR/m5out-scalar-u74-lowmem" --remote-gdb-port 0 \
    "$PIPELINE_DIR/configs/se_config_u74_lowmem.py" \
    --binary "$SCALAR_BIN" --clk-freq "$CLK_FREQ"

STATS="$PIPELINE_DIR/m5out-scalar-u74-lowmem/stats.txt"
if [ ! -s "$STATS" ]; then
    echo
    echo "FAILED: $STATS is missing or empty. Check for a traceback above --"
    echo "if RISCVMatchedCacheHierarchy's import failed, the script should have"
    echo "printed a fallback message instead of dying silently; if you don't see"
    echo "that message either, something else is wrong with U74Processor itself."
    exit 1
fi

echo "Done. stats.txt and config.json are in:"
echo "  $PIPELINE_DIR/m5out-scalar-u74-lowmem/"
echo
echo "Convert with (using the U74 template -- real core structure is the same,"
echo "the memory-size STATIC ASSUMPTION note in that template now applies for"
echo "real, since this run's memory system genuinely isn't the real board's):"
echo "  python3 mcpat/gem5_to_mcpat.py --stats m5out-scalar-u74-lowmem/stats.txt \\"
echo "      --clk-freq 1200MHz --l1i 32KiB --l1d 32KiB \\"
echo "      --template mcpat/template_u74_riscv.xml \\"
echo "      --out mcpat-scalar-u74-lowmem.xml"
