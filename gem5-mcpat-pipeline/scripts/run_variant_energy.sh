#!/usr/bin/env bash
# gem5 MinorCPU + McPAT energy for one already-built variant folder.
# Usage: bash run_variant_energy.sh <VARIANT_DIR> <tag>
# Requires: the variant already has build/bench and build/bench-scalar.
# Run with conda OFF. Basis: 100 MHz, VLEN 256, 22 nm, fp 0.60, 16 KiB L1.
set -e
VAR="$1"; TAG="$2"
PIPE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PIPE"
REPO_DIR="$VAR" ./scripts/run_both.sh
for t in rvv scalar; do
  python3 mcpat/gem5_to_mcpat.py --stats m5out-$t/stats.txt --clk-freq 100MHz \
      --l1i 16KiB --l1d 16KiB --fp-fraction 0.60 \
      --template mcpat/template_inorder_riscv.xml --out mcpat-$t.xml >/dev/null 2>&1
  ~/mcpat/mcpat -infile mcpat-$t.xml -print_level 5 > mcpat-$t-out.txt 2>/dev/null
  s=$(grep -m1 '^simSeconds' m5out-$t/stats.txt|awk '{print $2}')
  c=$(grep -m1 'committedInstType::total' m5out-$t/stats.txt|awk '{print $2}')
  P=$(grep -m1 'Runtime Dynamic =' mcpat-$t-out.txt|awk '{print $4}')
  echo "RESULT $TAG $t: committed=$c time=${s}s P=${P}W energy=$(awk "BEGIN{printf \"%.4f mJ\",$P*$s*1000}")"
done
