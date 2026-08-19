#!/usr/bin/env bash
# Clones and builds gem5. This step is the slow one -- expect anywhere from
# 20 minutes to well over an hour, depending on your machine and how many
# cores scons can use.
set -euo pipefail

GEM5_DIR="${GEM5_DIR:-$HOME/gem5}"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "$GEM5_DIR" ]; then
    git clone https://github.com/gem5/gem5.git "$GEM5_DIR"
fi
cd "$GEM5_DIR"
git checkout stable

if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt --break-system-packages || \
        pip3 install -r requirements.txt --user
fi

# RISCV-only target is enough for this pipeline and builds faster than ALL.
# The gem5 team's own RVV example script uses "build/ALL/gem5.opt" -- if
# something about RVV support looks missing from this narrower build,
# rebuild with: scons build/ALL/gem5.opt -j$JOBS
scons build/RISCV/gem5.opt -j"$JOBS"

echo
echo "Built: $GEM5_DIR/build/RISCV/gem5.opt"
echo "Next: run 03_build_mcpat.sh, then set GEM5_BIN to the path above for scripts/run_both.sh"
