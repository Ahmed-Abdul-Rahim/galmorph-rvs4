#!/usr/bin/env bash
# Converts both gem5 runs to McPAT XML, then runs McPAT on each.
# Run this after scripts/run_both.sh has produced m5out-rvv/ and m5out-scalar/.
set -euo pipefail

MCPAT_BIN="${MCPAT_BIN:-$HOME/mcpat/mcpat}"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLK_FREQ="${CLK_FREQ:-100MHz}"
L1I="${L1I:-16KiB}"
L1D="${L1D:-16KiB}"

if [ ! -x "$MCPAT_BIN" ]; then
    echo "McPAT binary not found or not executable: $MCPAT_BIN"
    echo "Build it first with 03_build_mcpat.sh, or set MCPAT_BIN."
    exit 1
fi

for tag in rvv scalar; do
    STATS="$PIPELINE_DIR/m5out-$tag/stats.txt"
    if [ ! -f "$STATS" ]; then
        echo "Missing $STATS -- run scripts/run_both.sh first."
        exit 1
    fi
    echo "== Converting $tag =="
    python3 "$PIPELINE_DIR/mcpat/gem5_to_mcpat.py" \
        --stats "$STATS" \
        --clk-freq "$CLK_FREQ" --l1i "$L1I" --l1d "$L1D" \
        --template "$PIPELINE_DIR/mcpat/template_inorder_riscv.xml" \
        --out "$PIPELINE_DIR/mcpat-$tag.xml"

    echo "== Running McPAT on $tag =="
    "$MCPAT_BIN" -infile "$PIPELINE_DIR/mcpat-$tag.xml" -print_level 5 \
        > "$PIPELINE_DIR/mcpat-$tag-out.txt"
    echo "Wrote $PIPELINE_DIR/mcpat-$tag-out.txt"
done

echo
echo "Both reports are ready. Summarize them with:"
echo "  python3 scripts/compare_report.py"
