#!/usr/bin/env python3
"""
compare_report.py -- pulls the top-level Processor numbers out of both
McPAT text reports, pairs them with each run's simulated execution time,
and prints an RVV-vs-scalar table (also written to comparison.md).

McPAT's own output format has not changed field names in a long time, but
this script only reads the FIRST occurrence of each field in the file,
which is the top-level "Processor:" summary in a normal -print_level run.
If your local McPAT's output layout differs, open the .txt file and check
by eye -- the numbers are all there either way, this script just saves you
scrolling.

Usage: run from the pipeline directory after scripts/run_mcpat.sh:
    python3 scripts/compare_report.py
"""

import re
import sys
from pathlib import Path

FIELDS = [
    ("Area", r"Area\s*=\s*([\d.eE+-]+)\s*mm\^2"),
    ("Peak Dynamic", r"Peak Dynamic\s*=\s*([\d.eE+-]+)\s*W"),
    ("Subthreshold Leakage", r"Subthreshold Leakage\s*=\s*([\d.eE+-]+)\s*W"),
    ("Gate Leakage", r"Gate Leakage\s*=\s*([\d.eE+-]+)\s*W"),
    ("Runtime Dynamic", r"Runtime Dynamic\s*=\s*([\d.eE+-]+)\s*W"),
]

# Derived fields computed in this script rather than read from McPAT.
# Energy = Runtime Dynamic power x simSeconds (this is the convention
# already used for the numbers in the top-level README/report -- it
# intentionally excludes static leakage, matching that existing analysis).
DERIVED_LABELS = ["Sim Exec Time (s)", "Energy / inference (mJ)"]


def parse_report(path):
    text = Path(path).read_text()
    out = {}
    for label, pattern in FIELDS:
        m = re.search(pattern, text)
        out[label] = float(m.group(1)) if m else None
    return out


def parse_sim_seconds(stats_path):
    """
    Reads gem5's `simSeconds` stat: the simulated pipeline's execution
    time, derived from simulated cycle count / clock frequency inside
    gem5's own event-driven scheduler. This is host-independent -- the
    same binary + gem5 build + config produces the same simSeconds on
    any machine.

    Deliberately NOT used here: gem5's `hostSeconds` stat (the real
    wall-clock time gem5 itself took to run) or any external `time`
    wrapper around run_both.sh. Both of those measure how fast the host
    machine is, not the simulated core, so they differ between machines
    even for an identical run -- swapping either in here would silently
    make "Energy / inference" stop matching across systems.
    """
    if not stats_path.exists():
        return None
    text = stats_path.read_text()
    m = re.search(r"^simSeconds\s+([\d.eE+-]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def main():
    here = Path(__file__).resolve().parent.parent
    rvv_path = here / "mcpat-rvv-out.txt"
    scalar_path = here / "mcpat-scalar-out.txt"
    for p in (rvv_path, scalar_path):
        if not p.exists():
            sys.exit(f"missing {p} -- run scripts/run_mcpat.sh first")

    results = {}
    for tag, report_path in (("rvv", rvv_path), ("scalar", scalar_path)):
        vals = parse_report(report_path)
        sim_s = parse_sim_seconds(here / f"m5out-{tag}" / "stats.txt")
        vals["Sim Exec Time (s)"] = sim_s

        dynamic = vals.get("Runtime Dynamic")
        if dynamic is not None and sim_s is not None:
            vals["Energy / inference (mJ)"] = dynamic * sim_s * 1000
        else:
            vals["Energy / inference (mJ)"] = None

        results[tag] = vals

    rvv, scalar = results["rvv"], results["scalar"]
    all_labels = [label for label, _ in FIELDS] + DERIVED_LABELS

    lines = ["| Metric | RVV build | Scalar build | Delta (RVV - scalar) |",
             "|---|---|---|---|"]
    print(f"{'Metric':<26}{'RVV':>16}{'Scalar':>16}{'Delta':>16}")
    for label in all_labels:
        rv, sc = rvv.get(label), scalar.get(label)
        if rv is None or sc is None:
            row = f"{label:<26}{'n/a':>16}{'n/a':>16}{'n/a':>16}"
            lines.append(f"| {label} | n/a | n/a | n/a |")
        else:
            delta = rv - sc
            row = f"{label:<26}{rv:>16.6f}{sc:>16.6f}{delta:>16.6f}"
            lines.append(f"| {label} | {rv:.6f} | {sc:.6f} | {delta:+.6f} |")
        print(row)

    if rvv.get("Sim Exec Time (s)") is None or scalar.get("Sim Exec Time (s)") is None:
        print(
            "\nNote: simSeconds was not found in one or both stats.txt files, "
            "so Sim Exec Time / Energy are n/a. Run scripts/find_stats.sh on "
            "the stats.txt in question to check the stat name for your gem5 "
            "version."
        )

    out_path = here / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
