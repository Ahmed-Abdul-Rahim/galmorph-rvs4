#!/usr/bin/env bash
# Runs build/bench (RVV) and build/bench-scalar under RiscvO3CPU.
# Deliberately a SEPARATE script from run_both.sh, not a modification to
# it -- your existing run_both.sh / run_mcpat.sh / comparison.md workflow
# for MinorCPU is untouched and keeps working exactly as it does today.
# This writes to differently-named output dirs (m5out-*-o3) so nothing
# here can collide with or overwrite your existing MinorCPU results.
#
# Usage: ./scripts/run_o3.sh
set -euo pipefail

GEM5_BIN="${GEM5_BIN:-$HOME/gem5/build/RISCV/gem5.opt}"
REPO_DIR="${REPO_DIR:-$HOME/galmorph-rvs4-main}"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLEN="${VLEN:-256}"
ELEN="${ELEN:-32}"
CLK_FREQ="${CLK_FREQ:-100MHz}"

RVV_BIN="$REPO_DIR/build/bench"
SCALAR_BIN="$REPO_DIR/build/bench-scalar"

for f in "$GEM5_BIN" "$RVV_BIN" "$SCALAR_BIN"; do
    if [ ! -f "$f" ]; then
        echo "Missing file: $f"
        exit 1
    fi
done

echo "== Running RVV build (cpu=o3) =="
echo "   o3 + RVV is what gem5's own official RVV example uses, so this is"
echo "   the better-tested path for the RVV binary if MinorCPU gave you"
echo "   trouble here -- still worth watching this run's own stderr closely."
"$GEM5_BIN" --outdir "$PIPELINE_DIR/m5out-rvv-o3" --remote-gdb-port 0 \
    "$PIPELINE_DIR/configs/se_config.py" \
    --binary "$RVV_BIN" --cpu o3 --vlen "$VLEN" --elen "$ELEN" --clk-freq "$CLK_FREQ"

echo "== Running scalar build (cpu=o3) =="
"$GEM5_BIN" --outdir "$PIPELINE_DIR/m5out-scalar-o3" --remote-gdb-port 0 \
    "$PIPELINE_DIR/configs/se_config.py" \
    --binary "$SCALAR_BIN" --cpu o3 --vlen "$VLEN" --elen "$ELEN" --clk-freq "$CLK_FREQ"

echo "Done. stats.txt and config.json are in:"
echo "  $PIPELINE_DIR/m5out-rvv-o3/"
echo "  $PIPELINE_DIR/m5out-scalar-o3/"
echo
echo "Before converting to McPAT: run scripts/find_stats.sh against"
echo "m5out-rvv-o3/stats.txt and check its 'OoO:' sections against the"
echo "rob_*/rename_*/iq_* candidates in mcpat/gem5_to_mcpat.py's STAT_MAP --"
echo "those were inferred from this repo's confirmed board.processor.cores.core.*"
echo "naming, not confirmed against a real O3 stats.txt yet."
echo
echo "Convert with:"
echo "  python3 mcpat/gem5_to_mcpat.py --stats m5out-rvv-o3/stats.txt \\"
echo "      --clk-freq $CLK_FREQ --l1i 16KiB --l1d 16KiB \\"
echo "      --template mcpat/template_ooo_riscv.xml --out mcpat-rvv-o3.xml"
