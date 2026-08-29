"""
se_config_u74_lowmem.py -- the real U74 core, without the real board's
16GB memory reservation that's failing on your machine.

WHY THIS EXISTS
----------------
RISCVMatchedBoard's own source (gem5.prebuilt.riscvmatched.riscvmatched_board)
constructs itself from three independent pieces:
    cache_hierarchy = RISCVMatchedCacheHierarchy(l2_size=l2_size)
    memory          = U74Memory()          # <- reserves a fixed 16GiB
                                            #    range [0:0x400000000],
                                            #    matching the real HiFive
                                            #    Unmatched's physical
                                            #    address decode window.
                                            #    This is what's failing to
                                            #    mmap on a 16GB host.
    processor       = U74Processor(is_fs=is_fs)  # <- the actual real,
                                            #    gem5-team-calibrated U74
                                            #    core. Independently
                                            #    importable, no
                                            #    dependency on U74Memory.
This file uses U74Processor and RISCVMatchedCacheHierarchy (the real core
and real cache config) but swaps U74Memory for SingleChannelDDR3_1600 (the
same small memory component se_config.py already uses successfully for
MinorCPU/O3CPU) via a plain SimpleBoard, instead of RISCVMatchedBoard.

WHAT THIS CHANGES, HONESTLY
-----------------------------
You keep: the real U74 core's timing model (fetch/decode/issue widths,
pipeline behavior) -- this is the part that actually drives what McPAT
reports for the CPU, and it's identical to what RISCVMatchedBoard would
have given you. You also keep the real L1/L2 cache config via
RISCVMatchedCacheHierarchy.

You lose: the real board's actual DRAM controller/interface model and its
real 16GB address decode window. Your report should say exactly that --
"the real, gem5-calibrated U74 core, with an invented smaller memory
system substituted for the real board's DRAM interface, due to a 16GB
host memory constraint" -- not "the real HiFive Unmatched board." This is
a smaller, disclosed compromise, not the full board -- same honesty
standard as every STATIC ASSUMPTION marker in the McPAT templates.

VERIFY BEFORE TRUSTING THIS FILE
-----------------------------------
Built from reading RISCVMatchedBoard's actual source (confirmed via a
real gem5.googlesource.com diff), not from running it -- I don't have a
gem5 build to test against. Confirm these two imports exist in your
checkout before relying on this:
    python3 -c "from gem5.prebuilt.riscvmatched.riscvmatched_processor import U74Processor; print('OK')"
    # (run through gem5.opt -c, not plain python3 -- see the note in
    # se_config_u74.py about m5.objects only existing inside gem5's own
    # interpreter)
If RISCVMatchedCacheHierarchy's import path below doesn't match your
checkout, fall back to PrivateL1CacheHierarchy (already proven working
elsewhere in this pipeline) instead -- you lose the real L2 config but
keep the real core, which is the more important piece.

Usage
-----
    <gem5-build>/build/RISCV/gem5.opt --outdir <outdir> --remote-gdb-port 0 \\
        configs/se_config_u74_lowmem.py --binary <path-to-bench-scalar>

Same scalar-only restriction as se_config_u74.py -- U74 has no RVV support.
"""

import argparse

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

# Verify these two against your checkout -- see the module docstring.
from gem5.prebuilt.riscvmatched.riscvmatched_processor import U74Processor
try:
    from gem5.prebuilt.riscvmatched.riscvmatched_cache import RISCVMatchedCacheHierarchy
    _HAVE_REAL_CACHE = True
except ImportError:
    from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import (
        PrivateL1CacheHierarchy,
    )
    _HAVE_REAL_CACHE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True,
                         help="path to build/bench-scalar -- U74 has no RVV, "
                              "do not use build/bench")
    parser.add_argument("--clk-freq", default="1.2GHz",
                         help="matches gem5's own RISCVMatchedBoard default")
    parser.add_argument("--l2-size", default="2MiB",
                         help="real FU740 L2 size, only used if "
                              "RISCVMatchedCacheHierarchy imports successfully")
    args = parser.parse_args()

    if _HAVE_REAL_CACHE:
        cache_hierarchy = RISCVMatchedCacheHierarchy(l2_size=args.l2_size)
        print(f"Using the real RISCVMatchedCacheHierarchy (l2_size={args.l2_size})")
    else:
        cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="32KiB", l1i_size="32KiB")
        print("RISCVMatchedCacheHierarchy import failed -- fell back to a plain "
              "32KiB/32KiB PrivateL1CacheHierarchy. Real L2 config lost; real "
              "core (U74Processor) still in use. Fix the import if the real "
              "cache config matters for your report.")

    # The actual fix: small memory instead of U74Memory()'s 16GB reservation.
    memory = SingleChannelDDR3_1600()

    processor = U74Processor(is_fs=False)

    board = SimpleBoard(
        clk_freq=args.clk_freq,
        processor=processor,
        memory=memory,
        cache_hierarchy=cache_hierarchy,
    )

    binary = BinaryResource(local_path=args.binary, architecture=ISA.RISCV)
    board.set_se_binary_workload(binary)

    simulator = Simulator(board=board)
    print(f"Beginning U74 (real core, small memory) simulation: binary={args.binary}")
    print("Reminder: scalar-only, real U74Processor, NOT the real U74Memory --")
    print("see this file's docstring for exactly what that trade-off means.")
    simulator.run()
    print(f"Done. Exit reason: {simulator.get_last_exit_event_cause()}")


if __name__ == "__main__":
    main()
