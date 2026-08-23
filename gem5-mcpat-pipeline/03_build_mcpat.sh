#!/usr/bin/env bash
set -euo pipefail

MCPAT_DIR="${MCPAT_DIR:-$HOME/mcpat}"
# Pinned for the same reason as gem5's GEM5_REF: a moving default branch
# means two machines can silently build against different McPAT internals.
# v1.3.0 is McPAT's latest tagged release (currently identical to master).
MCPAT_REF="${MCPAT_REF:-v1.3.0}"

if [ ! -d "$MCPAT_DIR" ]; then
    git clone https://github.com/HewlettPackard/mcpat.git "$MCPAT_DIR"
fi
cd "$MCPAT_DIR"

if [ ! -d .git ]; then
    echo "MCPAT_DIR ($MCPAT_DIR) exists but is not a git checkout -- refusing"
    echo "to run git commands against it. Point MCPAT_DIR at an empty"
    echo "directory or a real mcpat clone."
    exit 1
fi

git fetch --tags
if ! git checkout "$MCPAT_REF"; then
    echo
    echo "Could not check out $MCPAT_REF. If you have local edits to this"
    echo "mcpat checkout, commit or stash them first, then re-run this script."
    exit 1
fi

make -j"$(nproc)"

echo
echo "Built: $MCPAT_DIR/mcpat  (pinned to $MCPAT_REF)"
echo
echo "Before your first real run, open a sample file in this checkout, e.g.:"
echo "  $MCPAT_DIR/ProcessorDescriptionFiles/ARM_A9_2GHz.xml"
echo "and compare its component/param structure against"
echo "  mcpat/template_inorder_riscv.xml"
echo "from this pipeline, since no RISC-V sample ships with McPAT itself."
