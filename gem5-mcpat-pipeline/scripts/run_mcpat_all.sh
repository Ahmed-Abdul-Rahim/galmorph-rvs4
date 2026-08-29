#!/usr/bin/env bash
# Converts and runs McPAT for every architecture variant you've actually
# produced results for (auto-detected -- you don't need to have run all
# three). Picks the matching McPAT template per architecture:
#   minor (your existing scripts/run_both.sh)  -> mcpat/template_inorder_riscv.xml
#   o3    (scripts/run_o3.sh)                  -> mcpat/template_ooo_riscv.xml
#   u74   (scripts/run_u74.sh)                 -> mcpat/template_u74_riscv.xml
#
# Directory naming note: this intentionally does NOT assume run_both.sh
# tags its output by CPU -- your run_both.sh is untouched and still
# writes m5out-rvv/m5out-scalar exactly as it always has (so your
# existing scripts/run_mcpat.sh and comparison.md workflow keeps working
# unmodified). Only the NEW architectures (o3, u74) get their own
# distinctly-named dirs, from the NEW scripts (run_o3.sh, run_u74.sh) --
# nothing here can collide with what you already have.
#
# Run this after scripts/run_both.sh, scripts/run_o3.sh, and/or
# scripts/run_u74.sh (any subset). Safe to run multiple times as you add
# more architectures -- only converts what it finds, and never overwrites
# your existing mcpat-rvv.xml/mcpat-scalar.xml (minor's output keeps its
# original filenames, matching what scripts/compare_report.py already expects).
#
# Usage: ./scripts/run_mcpat_all.sh
set -uo pipefail

MCPAT_BIN="${MCPAT_BIN:-$HOME/mcpat/mcpat}"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLK_FREQ="${CLK_FREQ:-100MHz}"     # for minor/o3 -- your invented core's clock
L1I="${L1I:-16KiB}"
L1D="${L1D:-16KiB}"

if [ ! -x "$MCPAT_BIN" ]; then
    echo "McPAT binary not found or not executable: $MCPAT_BIN"
    echo "Build it first with 03_build_mcpat.sh, or set MCPAT_BIN."
    exit 1
fi

CONVERTED=0

convert_and_run () {
    local tag="$1" stats_dir="$2" template="$3" clk="$4" l1i="$5" l1d="$6"
    local stats="$stats_dir/stats.txt"
    if [ ! -f "$stats" ]; then
        return
    fi
    echo "== Converting $tag ($template, clk=$clk, l1i=$l1i, l1d=$l1d) =="
    python3 "$PIPELINE_DIR/mcpat/gem5_to_mcpat.py" \
        --stats "$stats" \
        --clk-freq "$clk" --l1i "$l1i" --l1d "$l1d" \
        --template "$PIPELINE_DIR/mcpat/$template" \
        --out "$PIPELINE_DIR/mcpat-$tag.xml"
    if [ $? -ne 0 ]; then
        echo "  Conversion failed for $tag, skipping McPAT run."
        return
    fi
    echo "== Running McPAT on $tag =="
    "$MCPAT_BIN" -infile "$PIPELINE_DIR/mcpat-$tag.xml" -print_level 5 \
        > "$PIPELINE_DIR/mcpat-$tag-out.txt"
    if [ $? -ne 0 ]; then
        echo "  McPAT itself failed for $tag -- check $PIPELINE_DIR/mcpat-$tag-out.txt"
        return
    fi
    echo "Wrote $PIPELINE_DIR/mcpat-$tag-out.txt"
    echo
    CONVERTED=$((CONVERTED+1))
}

# minor: your existing, already-validated run_both.sh output. Filenames
# match what scripts/run_mcpat.sh and scripts/compare_report.py already
# use -- if you've already run scripts/run_mcpat.sh for this, these two
# conversions just reproduce the same mcpat-rvv.xml/mcpat-scalar.xml
# byte-for-byte (nothing about minor's behavior changed).
convert_and_run "rvv"    "$PIPELINE_DIR/m5out-rvv"    "template_inorder_riscv.xml" "$CLK_FREQ" "$L1I" "$L1D"
convert_and_run "scalar" "$PIPELINE_DIR/m5out-scalar" "template_inorder_riscv.xml" "$CLK_FREQ" "$L1I" "$L1D"

# o3: from scripts/run_o3.sh
convert_and_run "rvv-o3"    "$PIPELINE_DIR/m5out-rvv-o3"    "template_ooo_riscv.xml" "$CLK_FREQ" "$L1I" "$L1D"
convert_and_run "scalar-o3" "$PIPELINE_DIR/m5out-scalar-o3" "template_ooo_riscv.xml" "$CLK_FREQ" "$L1I" "$L1D"

# u74: from scripts/run_u74.sh. Real board numbers, not the invented-core
# flags above -- see configs/se_config_u74.py and template_u74_riscv.xml.
convert_and_run "scalar-u74" "$PIPELINE_DIR/m5out-scalar-u74" "template_u74_riscv.xml" "1200MHz" "32KiB" "32KiB"

if [ "$CONVERTED" -eq 0 ]; then
    echo "No m5out*/stats.txt found anywhere expected. Run scripts/run_both.sh,"
    echo "scripts/run_o3.sh, and/or scripts/run_u74.sh first."
    exit 1
fi

echo "Converted and ran McPAT for $CONVERTED result set(s)."
echo "Compare them by reading the mcpat-*-out.txt files directly, or extend"
echo "scripts/compare_report.py to loop over the o3/u74-tagged filenames"
echo "the same way this script does."
