#!/usr/bin/env python3
"""
gem5_to_mcpat.py -- fills template_inorder_riscv.xml from a gem5 stats.txt.

Why a hand-written mapping instead of a generic parser: gem5 stat names
shift a little between versions and CPU models (MinorCPU, TimingSimpleCPU,
and AtomicSimpleCPU do not all report the same things). Run
scripts/find_stats.sh against your real stats.txt first, and compare what
it prints against STAT_MAP below. Fix any entry that does not match.

Every value this script writes into the McPAT XML falls into one of
three groups, and it tells you which group as it runs:
  - measured   : read straight from stats.txt under the given key.
  - fallback   : the key was not found, so a documented estimate was
                 used instead. Printed as a warning. Fine for a first
                 pass; replace with a real number if you need precision.
  - fixed      : a constant that does not come from gem5 at all
                 (clock rate, cache size) -- these come from the same
                 flags you passed to se_config.py, given again here so
                 both files agree.

Usage
-----
    python3 gem5_to_mcpat.py \
        --stats ../m5out-rvv/stats.txt \
        --clk-freq 100MHz --l1i 16KiB --l1d 16KiB \
        --template template_inorder_riscv.xml \
        --out mcpat-rvv.xml
"""

import argparse
import re
import sys

# gem5 stat key candidates, in the order to try them. The first one found
# in stats.txt wins. "high" entries fail the run loudly if none match,
# because the template cannot proceed without them. "soft" entries fall
# back to an estimate (see FALLBACKS) and only print a warning.
STAT_MAP = {
    # confidence: high -- these names have been stable across gem5 CPU
    # models for a long time.
    "total_cycles":        {"confidence": "high", "keys": ["system.cpu.numCycles", "board.processor.cores.core.numCycles"]},
    "total_instructions":  {"confidence": "high", "keys": ["system.cpu.numInsts", "system.cpu.committedInsts", "board.processor.cores.core.commitStats0.committedInstType::total", "board.processor.cores.core.commitStats0.numInsts"]},

    # confidence: soft -- naming varies by CPU model/version; verify with
    # find_stats.sh and add the exact key you see if it is missing here.
    "idle_cycles":          {"confidence": "soft", "keys": ["system.cpu.idleCycles", "system.cpu.numCycles::idle"]},
    "branch_instructions":  {"confidence": "soft", "keys": ["system.cpu.commit.branches", "system.cpu.branchPred.branches", "system.cpu.branchPred.lookups", "board.processor.cores.core.branchPred.committed_0::total"]},
    "branch_mispredictions": {"confidence": "soft", "keys": ["system.cpu.branchPred.mispredicted", "system.cpu.branchPred.condIncorrect", "system.cpu.commit.branchMispredicts", "board.processor.cores.core.branchPred.mispredicted_0::total"]},
    "load_instructions":    {"confidence": "soft", "keys": ["system.cpu.commit.loads", "system.cpu.iew.iewExecLoadInsts", "system.cpu.dcache.ReadReq.accesses::total", "board.cache_hierarchy.l1dcaches.ReadReq.accesses::total"]},
    "store_instructions":   {"confidence": "soft", "keys": ["system.cpu.commit.stores", "system.cpu.dcache.WriteReq.accesses::total", "board.cache_hierarchy.l1dcaches.WriteReq.accesses::total"]},
    "icache_accesses":      {"confidence": "soft", "keys": ["system.cpu.icache.overallAccesses::total", "system.cpu.icache.ReadReq.accesses::total", "board.cache_hierarchy.l1icaches.overallAccesses::total"]},
    "icache_misses":        {"confidence": "soft", "keys": ["system.cpu.icache.overallMisses::total", "system.cpu.icache.ReadReq.misses::total", "board.cache_hierarchy.l1icaches.overallMisses::total"]},
    "dcache_read_misses":   {"confidence": "soft", "keys": ["system.cpu.dcache.ReadReq.misses::total", "board.cache_hierarchy.l1dcaches.ReadReq.misses::total"]},
    "dcache_write_misses":  {"confidence": "soft", "keys": ["system.cpu.dcache.WriteReq.misses::total", "board.cache_hierarchy.l1dcaches.WriteReq.misses::total"]},
}

# Used only when a "soft" key above is not found in stats.txt. Each is a
# function of the already-resolved values dict. Keep these conservative
# and clearly separate from measured numbers -- see the module docstring.
FALLBACKS = {
    "idle_cycles":           lambda v: 0.0,  # single-threaded, compute-bound SE run: assume no idle time
    "branch_instructions":   lambda v: 0.15 * v["total_instructions"],
    "branch_mispredictions": lambda v: 0.05 * v["branch_instructions"],
    "load_instructions":     lambda v: 0.20 * v["total_instructions"],
    "store_instructions":    lambda v: 0.10 * v["total_instructions"],
    "icache_accesses":       lambda v: v["total_instructions"],
    "icache_misses":         lambda v: 0.0,
    "dcache_read_misses":    lambda v: 0.0,
    "dcache_write_misses":   lambda v: 0.0,
}


def parse_stats_txt(path):
    stats = {}
    line_re = re.compile(r"^(\S+)\s+(\S+)")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            m = line_re.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            try:
                stats[key] = float(val)
            except ValueError:
                continue  # non-numeric stat (e.g. a string-valued stat); skip
    return stats


def resolve(name, stats, values, errors, warnings):
    spec = STAT_MAP[name]
    for key in spec["keys"]:
        if key in stats:
            values[name] = stats[key]
            return
    if spec["confidence"] == "high":
        errors.append(
            f"Required stat '{name}' not found. Tried: {spec['keys']}. "
            f"Run scripts/find_stats.sh on your stats.txt, find the real "
            f"name, and add it to STAT_MAP['{name}']['keys'] in this script."
        )
        values[name] = 0.0
        return
    # soft: fall back
    fb = FALLBACKS[name](values)
    values[name] = fb
    warnings.append(f"'{name}' not found (tried {spec['keys']}); used a fallback estimate: {fb:.0f}")


def parse_size_to_bytes(s):
    s = s.strip()
    m = re.match(r"([0-9.]+)\s*([A-Za-z]*)", s)
    if not m:
        raise ValueError(f"Cannot parse size: {s}")
    num, unit = float(m.group(1)), m.group(2).lower()
    mult = {"": 1, "b": 1, "kib": 1024, "kb": 1000, "mib": 1024 * 1024, "mb": 1000000}
    if unit not in mult:
        raise ValueError(f"Unknown size unit '{unit}' in {s}")
    return int(num * mult[unit])


def parse_freq_to_mhz(s):
    s = s.strip()
    m = re.match(r"([0-9.]+)\s*([A-Za-z]*)", s)
    if not m:
        raise ValueError(f"Cannot parse frequency: {s}")
    num, unit = float(m.group(1)), m.group(2).lower()
    mult = {"hz": 1e-6, "khz": 1e-3, "mhz": 1, "ghz": 1000, "": 1}
    if unit not in mult:
        raise ValueError(f"Unknown frequency unit '{unit}' in {s}")
    return num * mult[unit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="path to gem5's stats.txt")
    ap.add_argument("--clk-freq", required=True, help="same value passed to se_config.py, e.g. 100MHz")
    ap.add_argument("--l1i", required=True, help="same value passed to se_config.py, e.g. 16KiB")
    ap.add_argument("--l1d", required=True, help="same value passed to se_config.py, e.g. 16KiB")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp-fraction", type=float, default=0.5,
                     help="fallback int/fp instruction split if no dynamic breakdown is found in stats.txt (0..1, share that is floating point)")
    args = ap.parse_args()

    stats = parse_stats_txt(args.stats)
    if not stats:
        sys.exit(f"error: read zero stats from {args.stats} -- is this a real gem5 stats.txt?")

    values = {}
    errors, warnings = [], []

    for name in ["total_cycles", "total_instructions"]:
        resolve(name, stats, values, errors, warnings)
    for name in ["idle_cycles", "branch_instructions", "branch_mispredictions",
                 "load_instructions", "store_instructions",
                 "icache_accesses", "icache_misses",
                 "dcache_read_misses", "dcache_write_misses"]:
        resolve(name, stats, values, errors, warnings)

    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    # derived values not looked up directly
    values["busy_cycles"] = max(values["total_cycles"] - values["idle_cycles"], 0.0)
    # no dynamic int/fp split found anywhere reliable across CPU models, so
    # this is always a fallback -- flag it plainly rather than pretend it's measured.
    values["fp_instructions"] = args.fp_fraction * values["total_instructions"]
    values["int_instructions"] = (1 - args.fp_fraction) * values["total_instructions"]
    warnings.append(
        f"'int_instructions'/'fp_instructions' split is a fixed fp-fraction={args.fp_fraction} "
        f"assumption, not measured -- this workload is float-heavy (S4D forward pass), adjust "
        f"--fp-fraction if you have a real ratio."
    )
    values["int_regfile_reads"] = 2 * values["int_instructions"]
    values["int_regfile_writes"] = values["int_instructions"]
    values["float_regfile_reads"] = 2 * values["fp_instructions"]
    values["float_regfile_writes"] = values["fp_instructions"]
    values["dcache_accesses"] = values["load_instructions"] + values["store_instructions"]

    # fixed, from the flags used with se_config.py
    values["clk_freq_mhz"] = parse_freq_to_mhz(args.clk_freq)
    values["l1i_bytes"] = parse_size_to_bytes(args.l1i)
    values["l1d_bytes"] = parse_size_to_bytes(args.l1d)

    with open(args.template) as f:
        xml = f.read()

    def sub(m):
        key = m.group(1)
        if key not in values:
            errors.append(f"template references GEM5:{key}, which this script does not compute")
            return "0"
        v = values[key]
        return str(int(v)) if float(v).is_integer() else f"{v:.6f}"

    xml_out = re.sub(r"GEM5:([A-Za-z0-9_]+)", sub, xml)

    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        f.write(xml_out)

    print(f"Wrote {args.out}")
    if warnings:
        print(f"\n{len(warnings)} value(s) used a fallback or fixed assumption instead of a measured stat:")
        for w in warnings:
            print("  -", w)


if __name__ == "__main__":
    main()
