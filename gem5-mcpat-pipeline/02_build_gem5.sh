#!/usr/bin/env bash
# Clones and builds gem5. This step is the slow one -- expect anywhere from
# 20 minutes to well over an hour, depending on your machine and how many
# cores scons can use.
#
# IMPORTANT: run this with any conda/venv environment DEACTIVATED. gem5's
# scons build links against the apt packages from 01_install_deps.sh
# (boost, protobuf, hdf5). conda in particular ships its own copies of
# those same libraries and has been found to conflict with gem5's build
# in this project -- if `conda info --envs` shows a starred environment,
# run `conda deactivate` first.
set -euo pipefail

GEM5_DIR="${GEM5_DIR:-$HOME/gem5}"
JOBS="${JOBS:-$(nproc)}"

# Pinned to the exact tagged release this pipeline has been built and
# validated against. "stable" is a moving branch -- two machines cloning
# on different days can silently get different timing-model internals,
# which shifts simSeconds/numCycles even though instruction counts still
# match. Pinning is what makes results reproducible across systems.
GEM5_REF="${GEM5_REF:-v25.1.0.1}"        # tag c8222cc67a399bfc01e8658dd14b30d5bfd634f9

if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    echo "A conda environment ('$CONDA_DEFAULT_ENV') is currently active."
    echo "Run 'conda deactivate' first -- conda's bundled libraries can"
    echo "conflict with gem5's build against the system boost/protobuf/hdf5"
    echo "installed by 01_install_deps.sh. (A plain venv is fine; conda is"
    echo "the specific thing that has caused problems here.)"
    exit 1
fi

if [ ! -d "$GEM5_DIR" ]; then
    git clone https://github.com/gem5/gem5.git "$GEM5_DIR"
fi
cd "$GEM5_DIR"

if [ ! -d .git ]; then
    echo "GEM5_DIR ($GEM5_DIR) exists but is not a git checkout -- refusing"
    echo "to run git commands against it. Point GEM5_DIR at an empty"
    echo "directory or a real gem5 clone."
    exit 1
fi

git fetch --tags
if ! git checkout "$GEM5_REF"; then
    echo
    echo "Could not check out $GEM5_REF. If you have local edits to this"
    echo "gem5 checkout, commit or stash them first, then re-run this script."
    exit 1
fi

# gem5's own requirements.txt (mypy, pre-commit) is dev tooling for
# contributing to gem5 itself -- it is NOT needed to build or run
# gem5.opt in SE mode, which is all this pipeline does. We deliberately
# do not install it here, and specifically never with pip's
# --break-system-packages flag: that flag overrides Ubuntu's
# externally-managed-environment guard and can put incompatible package
# versions where apt-managed tools expect their own, which can break
# other things on the system that depend on system Python. If you are
# doing gem5 core development and actually want mypy/pre-commit, install
# them into a dedicated (non-conda) venv, never into system Python.

# RISCV-only target is enough for this pipeline and builds faster than ALL.
# The gem5 team's own RVV example script uses "build/ALL/gem5.opt" -- if
# something about RVV support looks missing from this narrower build,
# rebuild with: scons build/ALL/gem5.opt -j$JOBS
scons build/RISCV/gem5.opt -j"$JOBS"

echo
echo "Built: $GEM5_DIR/build/RISCV/gem5.opt  (pinned to $GEM5_REF)"
echo "Next: run 03_build_mcpat.sh, then set GEM5_BIN to the path above for scripts/run_both.sh"
