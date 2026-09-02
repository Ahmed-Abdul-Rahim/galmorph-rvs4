#!/usr/bin/env bash
# Full benchmark suite: instruction counts (QEMU) for every S4D variant, then
# energy (gem5 + McPAT) on MinorCPU. Prerequisites in INSTALL.md must be done,
# and checkpoints present (scripts/fetch_checkpoints.sh). Run with conda OFF.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TC="riscv32-unknown-elf-gcc"; CF='-O2'
declare -A CFG=( [s4d-d64-native]=d64_native [s4d-d64-seq256]=d64_seq256 [s4d-d108-seq256]=d108_seq256 )

echo "==================== QEMU instruction counts ===================="
for d in "${!CFG[@]}"; do
  cd "$ROOT/variants/$d"
  python3 "$ROOT/scripts/bake_weights.py" --config "${CFG[$d]}" --ckpt-dir "$ROOT/checkpoints" >/dev/null
  make bench        CC=$TC CFLAGS="$CF" >/dev/null
  make bench-scalar CC=$TC CFLAGS="$CF" >/dev/null
  rvv=$(qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench        | awk '/TOTAL/{print $3}')
  sca=$(qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | awk '/TOTAL/{print $3}')
  pr=$(qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench | awk -F': ' '/Prediction/{print $2}')
  echo "  $d : RVV=$rvv  Scalar=$sca  pred=$pr"
done

echo "==================== Energy (gem5 MinorCPU + McPAT) ===================="
echo "Run each of these (conda OFF). Each prints two RESULT lines."
for d in "${!CFG[@]}"; do
  echo "  bash gem5-mcpat-pipeline/scripts/run_variant_energy.sh $ROOT/variants/$d ${CFG[$d]}"
done
