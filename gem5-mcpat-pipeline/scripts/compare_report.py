#!/usr/bin/env python3
"""
compare_report.py -- pulls the top-level Processor numbers out of both
McPAT text reports and prints them side by side.

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


def parse_report(path):
    text = Path(path).read_text()
    out = {}
    for label, pattern in FIELDS:
        m = re.search(pattern, text)
        out[label] = float(m.group(1)) if m else None
    return out


def main():
    here = Path(__file__).resolve().parent.parent
    rvv_path = here / "mcpat-rvv-out.txt"
    scalar_path = here / "mcpat-scalar-out.txt"
    for p in (rvv_path, scalar_path):
        if not p.exists():
            sys.exit(f"missing {p} -- run scripts/run_mcpat.sh first")

    rvv = parse_report(rvv_path)
    scalar = parse_report(scalar_path)

    lines = ["| Metric | RVV build | Scalar build | Delta (RVV - scalar) |",
             "|---|---|---|---|"]
    print(f"{'Metric':<24}{'RVV':>14}{'Scalar':>14}{'Delta':>14}")
    for label, _ in FIELDS:
        rv, sc = rvv[label], scalar[label]
        if rv is None or sc is None:
            row = f"{label:<24}{'n/a':>14}{'n/a':>14}{'n/a':>14}"
            lines.append(f"| {label} | n/a | n/a | n/a |")
        else:
            delta = rv - sc
            row = f"{label:<24}{rv:>14.4f}{sc:>14.4f}{delta:>14.4f}"
            lines.append(f"| {label} | {rv:.4f} | {sc:.4f} | {delta:+.4f} |")
        print(row)

    out_path = here / "comparison.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
