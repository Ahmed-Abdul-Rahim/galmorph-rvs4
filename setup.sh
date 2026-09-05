#!/usr/bin/env bash
# One command to set up and self-test every model, using only gcc and python3.
# After this runs, every model is built (host) and validated against an
# independent NumPy reference. Instruction counts and energy need more tools
# (RISC-V toolchain, gem5, McPAT); see README.md sections 5 and 6.
set -e
cd "$(dirname "$0")"
echo "==================== 1. checkpoints ===================="
bash scripts/fetch_checkpoints.sh
echo
echo "==================== 2. root model: S4D d108 ===================="
python3 scripts/bake_weights.py --config d108 --ckpt-dir checkpoints
make >/dev/null
python3 scripts/validate_oracle.py --config d108 --bin ./build/main --ckpt-dir checkpoints
echo
echo "==================== 3. variants ===================="
for pair in "s4d-d64-native:d64_native" "s4d-d64-seq256:d64_seq256" "s4d-d108-seq256:d108_seq256"; do
  d="${pair%%:*}"; cfg="${pair##*:}"
  echo "-- $d --"
  ( cd "variants/$d" \
    && python3 ../../scripts/bake_weights.py --config "$cfg" \
    && make >/dev/null \
    && python3 ../../scripts/validate_oracle.py --config "$cfg" --bin ./build/main )
done
echo
echo "Setup complete: all models built with gcc and validated to about 5e-05."
echo "Instruction counts and energy: install the RISC-V toolchain, gem5, and McPAT"
echo "(README section 5), then run:  bash scripts/run_all_benchmarks.sh"
