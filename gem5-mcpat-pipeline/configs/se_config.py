"""
se_config.py -- gem5 Syscall-Emulation config for the galmorph-rvs4 core.

This script runs ONE local RISC-V binary (your build/bench or
build/bench-scalar) in SE mode and writes stats.txt + config.json to
--outdir. Those two files are the input the McPAT converter needs.

It does not download anything from the network. It does not need an
operating-system image. This matches how you run the binary today
under qemu-riscv32.

Usage
-----
    <gem5-build>/build/RISCV/gem5.opt --outdir <outdir> \
        configs/se_config.py --binary <path-to-elf> \
        [--cpu minor|timing|atomic] [--vlen 256] [--elen 32] \
        [--clk-freq 100MHz] [--l1i 16KiB] [--l1d 16KiB]

Notes on the choices below
---------------------------
- Default CPU model is "minor". RiscvMinorCPU is gem5's pipelined
  in-order model. It is a closer match to a VeeR-class in-order core
  than RiscvTimingSimpleCPU, and it reports more of the per-structure
  activity counts that McPAT wants. The gem5 team's own worked RVV
  example uses the out-of-order RiscvO3CPU, so MinorCPU with RVV is
  a LESS-tested combination. If MinorCPU faults or hangs on the RVV
  binary, rerun with --cpu timing first to isolate whether the
  problem is RVV-in-general or RVV-in-MinorCPU specifically.
- clk-freq, l1i, and l1d are stand-in values for a small
  power-constrained embedded core. Change them to match your real
  target if you have numbers for it; they are plain command-line
  flags, not buried constants.
"""

import argparse

from m5.objects import RiscvMinorCPU, RiscvTimingSimpleCPU, RiscvAtomicSimpleCPU

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import (
    PrivateL1CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

CPU_CLASSES = {
    "minor": RiscvMinorCPU,
    "timing": RiscvTimingSimpleCPU,
    "atomic": RiscvAtomicSimpleCPU,
}


class RVVCore(BaseCPUCore):
    """One core, with VLEN/ELEN set the same way the official gem5 RVV
    example sets them (core.isa[0].vlen / .elen), so this works
    whichever CPU class we hand it."""

    def __init__(self, cpu_class, vlen, elen, cpu_id=0):
        super().__init__(core=cpu_class(cpu_id=cpu_id), isa=ISA.RISCV)
        self.core.isa[0].vlen = vlen
        self.core.isa[0].elen = elen
        self.core.isa[0].riscv_type = "RV32"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="path to build/bench or build/bench-scalar")
    parser.add_argument("--cpu", choices=CPU_CLASSES.keys(), default="minor")
    parser.add_argument("--vlen", type=int, default=256, help="matches VLEN in your Makefile/QEMU setup")
    parser.add_argument("--elen", type=int, default=32, help="matches ELEN in your Makefile/QEMU setup")
    parser.add_argument("--clk-freq", default="100MHz", help="core clock; a stand-in, change to your real target")
    parser.add_argument("--l1i", default="16KiB")
    parser.add_argument("--l1d", default="16KiB")
    args = parser.parse_args()

    cpu_class = CPU_CLASSES[args.cpu]

    cache_hierarchy = PrivateL1CacheHierarchy(l1d_size=args.l1d, l1i_size=args.l1i)
    memory = SingleChannelDDR3_1600()
    processor = BaseCPUProcessor(
        cores=[RVVCore(cpu_class, args.vlen, args.elen, cpu_id=0)]
    )

    board = SimpleBoard(
        clk_freq=args.clk_freq,
        processor=processor,
        memory=memory,
        cache_hierarchy=cache_hierarchy,
    )

    binary = BinaryResource(local_path=args.binary, architecture=ISA.RISCV)
    board.set_se_binary_workload(binary)

    simulator = Simulator(board=board)
    print(f"Beginning simulation: cpu={args.cpu} vlen={args.vlen} elen={args.elen} binary={args.binary}")
    simulator.run()
    print(f"Done. Exit reason: {simulator.get_last_exit_event_cause()}")


main()
