"""
diag_u74_lowmem.py -- maximally verbose, step-by-step diagnostic for the
se_config_u74_lowmem.py silent failure. Every step is wrapped so that
whatever is happening becomes visible -- an import error, an exception
during construction, or something being lost to output buffering.

Usage:
    <gem5-build>/build/RISCV/gem5.opt --outdir /tmp/u74_diag3 \
        diag_u74_lowmem.py --binary <path-to-bench-scalar>
"""
import argparse
import sys
import traceback


def step(name):
    print(f">>> STEP: {name}", flush=True)


def fail(name, exc):
    print(f">>> FAILED at: {name}", flush=True)
    print(">>> Exception:", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--clk-freq", default="1.2GHz")
parser.add_argument("--l2-size", default="2MiB")
step("argparse defined, about to parse")
try:
    args = parser.parse_args()
    step(f"args parsed OK: binary={args.binary} clk_freq={args.clk_freq}")
except SystemExit:
    raise
except Exception as e:
    fail("argparse.parse_args()", e)

step("importing gem5.components.boards.simple_board")
try:
    from gem5.components.boards.simple_board import SimpleBoard
except Exception as e:
    fail("import SimpleBoard", e)

step("importing gem5.components.memory")
try:
    from gem5.components.memory import SingleChannelDDR3_1600
except Exception as e:
    fail("import SingleChannelDDR3_1600", e)

step("importing gem5.isas")
try:
    from gem5.isas import ISA
except Exception as e:
    fail("import ISA", e)

step("importing gem5.resources.resource")
try:
    from gem5.resources.resource import BinaryResource
except Exception as e:
    fail("import BinaryResource", e)

step("importing gem5.simulate.simulator")
try:
    from gem5.simulate.simulator import Simulator
except Exception as e:
    fail("import Simulator", e)

step("importing U74Processor")
try:
    from gem5.prebuilt.riscvmatched.riscvmatched_processor import U74Processor
    step("U74Processor imported OK")
except Exception as e:
    fail("import U74Processor", e)

step("importing RISCVMatchedCacheHierarchy (may legitimately not exist -- not fatal)")
have_real_cache = False
try:
    from gem5.prebuilt.riscvmatched.riscvmatched_cache import RISCVMatchedCacheHierarchy
    have_real_cache = True
    step("RISCVMatchedCacheHierarchy imported OK")
except Exception as e:
    step(f"RISCVMatchedCacheHierarchy import failed ({type(e).__name__}: {e}) -- will use PrivateL1CacheHierarchy instead")
    try:
        from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import PrivateL1CacheHierarchy
    except Exception as e2:
        fail("import PrivateL1CacheHierarchy (fallback)", e2)

step("constructing cache_hierarchy")
try:
    if have_real_cache:
        cache_hierarchy = RISCVMatchedCacheHierarchy(l2_size=args.l2_size)
    else:
        cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="32KiB", l1i_size="32KiB")
    step("cache_hierarchy constructed OK")
except Exception as e:
    fail("construct cache_hierarchy", e)

step("constructing memory (SingleChannelDDR3_1600)")
try:
    memory = SingleChannelDDR3_1600()
    step("memory constructed OK")
except Exception as e:
    fail("construct memory", e)

step("constructing U74Processor")
try:
    processor = U74Processor(is_fs=False)
    step("U74Processor constructed OK")
except Exception as e:
    fail("construct U74Processor", e)

step("setting riscv_type=RV32 on U74Processor's cores (the same fix "
     "se_config.py's RVVCore already applies -- U74Processor defaults to "
     "RV64 since real U74 silicon is 64-bit; never carried this fix over "
     "to the U74 configs, which is very likely today's actual bug)")
try:
    cores = processor.cores
    step(f"processor.cores found, {len(cores)} core(s)")
    for i, c in enumerate(cores):
        c.core.isa[0].riscv_type = "RV32"
        step(f"core[{i}].core.isa[0].riscv_type set to RV32")
except Exception as e:
    print(">>> processor.cores/.core.isa[0] path didn't work as expected.", flush=True)
    print(">>> Introspecting the real object structure instead of guessing further:", flush=True)
    print(">>> dir(processor):", [a for a in dir(processor) if not a.startswith('_')], flush=True)
    fail("set riscv_type on U74Processor cores", e)

step("constructing SimpleBoard")
try:
    board = SimpleBoard(
        clk_freq=args.clk_freq,
        processor=processor,
        memory=memory,
        cache_hierarchy=cache_hierarchy,
    )
    step("SimpleBoard constructed OK")
except Exception as e:
    fail("construct SimpleBoard", e)

step("constructing BinaryResource")
try:
    binary = BinaryResource(local_path=args.binary, architecture=ISA.RISCV)
    step("BinaryResource constructed OK")
except Exception as e:
    fail("construct BinaryResource", e)

step("calling board.set_se_binary_workload(binary)")
try:
    board.set_se_binary_workload(binary)
    step("set_se_binary_workload OK")
except Exception as e:
    fail("set_se_binary_workload", e)

step("constructing Simulator(board=board)")
try:
    simulator = Simulator(board=board)
    step("Simulator constructed OK")
except Exception as e:
    fail("construct Simulator", e)

step("calling simulator.run() -- this is where the mmap crash happened last time")
try:
    simulator.run()
    step("simulator.run() returned normally")
except SystemExit as e:
    print(f">>> simulator.run() triggered SystemExit({e.code}) -- this may be gem5's own fatal() handler, not a Python exception", flush=True)
    raise
except Exception as e:
    fail("simulator.run()", e)

step(f"DONE. exit cause: {simulator.get_last_exit_event_cause()}")
