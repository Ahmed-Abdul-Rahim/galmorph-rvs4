"""
se_config_u74.py -- run build/bench-scalar on gem5's RISCVMatchedBoard,
a model of the real SiFive FU740 (the chip on the HiFive Unmatched
board), for a real-silicon cross-check on your invented core's numbers.

Usage
-----
    <gem5-build>/build/RISCV/gem5.opt --outdir <outdir> \
        configs/se_config_u74.py --binary <path-to-elf> [--clk-freq 1.2GHz]

READ THIS BEFORE RUNNING
-------------------------
1. SCALAR ONLY. Point --binary at build/bench-scalar. Do not point this
   at build/bench (the RVV binary) -- confirmed directly by gem5
   maintainers (github.com/orgs/gem5/discussions/523): the U74 model in
   gem5's stdlib does not support the RISC-V Vector extension. This
   script does not check for that mistake for you; it will simply run
   whatever ELF you give it and may fail confusingly on an RVV binary.

2. No --l1i/--l1d/--vlen/--elen flags here, unlike your other configs.
   That's deliberate, not an oversight: U74's cache sizes are fixed
   by the real chip (32KB L1I, 32KB L1D, per the FU740-C000 Owner's
   Manual), not something you get to choose the way you choose sizes
   for your own invented core. RISCVMatchedBoard exposes clk_freq and
   l2_size as constructor params (see below); it does not expose L1
   sizes as a param at all, because on the real chip they aren't a
   variable either.

3. This is a 64-bit core (RV64GC). Your build/bench-scalar is RV32.
   gem5's SE mode will run a 32-bit RISC-V static binary on a 64-bit
   RiscvISA core in principle (the ISA is a superset), but this
   combination is untested in this pipeline. If it faults, that's
   useful information in itself -- note it in your report rather than
   silently switching back, since "does the real chip's core model
   even accept our binary's ISA mode" is itself part of what "try
   running on U74" is meant to surface.

4. Verify RISCVMatchedBoard actually exists under this import path in
   YOUR gem5 checkout before trusting this file -- it was introduced in
   gem5 22.1:
       python3 -c "from gem5.prebuilt.riscvmatched.riscvmatched_board \
           import RISCVMatchedBoard; print('found it')"
   If that fails, your gem5 predates 22.1 or the module moved; check
       find $GEM5_DIR/src/python/gem5/prebuilt -iname '*matched*'
   and adjust the import below to match what you actually have.

WHEN YOU CONVERT THIS RUN'S stats.txt TO McPAT
------------------------------------------------
Use template_u74_riscv.xml, and pass these flags to gem5_to_mcpat.py
to match what actually ran (these are the real board's numbers, not
placeholders you're free to change):
    --clk-freq 1200MHz --l1i 32KiB --l1d 32KiB \
    --template mcpat/template_u74_riscv.xml
"""

import argparse

from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

# Verify this import path against your gem5 checkout -- see note 4 above.
from gem5.prebuilt.riscvmatched.riscvmatched_board import RISCVMatchedBoard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True,
                         help="path to build/bench-scalar -- see note 1, "
                              "U74 has no RVV support, do not use build/bench")
    parser.add_argument("--clk-freq", default="1.2GHz",
                         help="gem5's own RISCVMatchedBoard default, set from "
                              "the UC Davis HiFive Unmatched measurements")
    parser.add_argument("--l2-size", default="2MiB",
                         help="real FU740 shared L2 size, per SiFive's U74 "
                              "Core Complex Manual")
    args = parser.parse_args()

    board = RISCVMatchedBoard(
        clk_freq=args.clk_freq,
        l2_size=args.l2_size,
        is_fs=False,  # SE mode -- same mode your other configs use.
                      # SPEC CPU 2026's own gem5-based benchmark component
                      # uses RISCVMatchedBoard in this same SE-mode
                      # configuration, for reference.
    )

    binary = BinaryResource(local_path=args.binary, architecture=ISA.RISCV)
    board.set_se_binary_workload(binary)

    simulator = Simulator(board=board)
    print(f"Beginning U74 (real-silicon-calibrated) simulation: binary={args.binary}")
    print("Reminder: scalar-only. U74's gem5 model does not support RVV.")
    simulator.run()
    print(f"Done. Exit reason: {simulator.get_last_exit_event_cause()}")


if __name__ == "__main__":
    main()
