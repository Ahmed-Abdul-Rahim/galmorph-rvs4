#!/usr/bin/env bash
# Runs build/bench-scalar through gem5's RISCVMatchedBoard (real SiFive
# U74/FU740 model). SCALAR ONLY -- U74 has no RVV support in gem5, so
# there is no RVV counterpart to this script. See configs/se_config_u74.py
# for the full explanation.
#
# Usage: ./scripts/run_u74.sh
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

echo "== Running scalar build on U74 (RISCVMatchedBoard) =="
echo "   Real SiFive FU740 model -- 8-stage dual-issue in-order, 32KiB L1I/D."
echo "   This is the one architecture in this comparison with a documented"
echo "   real-silicon pedigree (see the deep-review doc), and the only one"
echo "   where you are NOT free to pick clk-freq/l1i/l1d -- they're fixed"
echo "   to match the real chip."
"$GEM5_BIN" --outdir "$PIPELINE_DIR/m5out-scalar-u74" --remote-gdb-port 0 \
    "$PIPELINE_DIR/configs/se_config_u74.py" \
    --binary "$SCALAR_BIN" --clk-freq "$CLK_FREQ"

STATS="$PIPELINE_DIR/m5out-scalar-u74/stats.txt"
if [ ! -s "$STATS" ]; then
    echo
    echo "FAILED: $STATS is missing or empty, even though gem5 exited without"
    echo "an error status. This usually means se_config_u74.py raised a Python"
    echo "exception before Root.instantiate() -- check for a traceback ABOVE"
    echo "this message (easy to miss if your terminal scrolled past it), and"
    echo "confirm config.json/config.ini also exist in that directory (if"
    echo "they don't either, the board never got instantiated at all)."
    exit 1
fi

echo "Done. stats.txt and config.json are in:"
echo "  $PIPELINE_DIR/m5out-scalar-u74/"
echo
echo "Convert with:"
echo "  python3 mcpat/gem5_to_mcpat.py --stats m5out-scalar-u74/stats.txt \\"
echo "      --clk-freq 1200MHz --l1i 32KiB --l1d 32KiB \\"
echo "      --template mcpat/template_u74_riscv.xml \\"
echo "      --out mcpat-scalar-u74.xml"
