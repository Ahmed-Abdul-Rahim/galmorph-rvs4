#!/usr/bin/env bash
set -euo pipefail

MCPAT_DIR="${MCPAT_DIR:-$HOME/mcpat}"

if [ ! -d "$MCPAT_DIR" ]; then
    git clone https://github.com/HewlettPackard/mcpat.git "$MCPAT_DIR"
fi
cd "$MCPAT_DIR"
make -j"$(nproc)"

echo
echo "Built: $MCPAT_DIR/mcpat"
echo
echo "Before your first real run, open a sample file in this checkout, e.g.:"
echo "  $MCPAT_DIR/ProcessorDescriptionFiles/ARM_A9_2GHz.xml"
echo "and compare its component/param structure against"
echo "  mcpat/template_inorder_riscv.xml"
echo "from this pipeline, since no RISC-V sample ships with McPAT itself."
