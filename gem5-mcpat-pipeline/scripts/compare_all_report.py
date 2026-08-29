#!/usr/bin/env python3
"""
compare_all_report.py -- summarizes McPAT results across every CPU
architecture you've actually run (minor, o3, u74), not just RVV-vs-scalar
for one architecture the way scripts/compare_report.py does.

Auto-detects which mcpat-*-out.txt files exist and only reports on those
-- you don't need to have run all three architectures. Does not touch or
overwrite comparison.md (that stays scripts/compare_report.py's file);
this writes comparison_all.md instead, so both can coexist.

Usage: run from the pipeline directory after scripts/run_mcpat_all.sh
(or scripts/run_mcpat.sh, or any subset):
    python3 scripts/compare_all_report.py
"""

import re
import sys
from pathlib import Path

FIELDS = [
    ("Area (mm^2)", r"Area\s*=\s*([\d.eE+-]+)\s*mm\^2"),
    ("Peak Dynamic (W)", r"Peak Dynamic\s*=\s*([\d.eE+-]+)\s*W"),
    ("Subthreshold Leakage (W)", r"Subthreshold Leakage\s*=\s*([\d.eE+-]+)\s*W"),
    ("Gate Leakage (W)", r"Gate Leakage\s*=\s*([\d.eE+-]+)\s*W"),
    ("Runtime Dynamic (W)", r"Runtime Dynamic\s*=\s*([\d.eE+-]+)\s*W"),
]
DERIVED_LABELS = ["Sim Exec Time (s)", "Energy / inference (mJ)"]
ALL_LABELS = [label for label, _ in FIELDS] + DERIVED_LABELS

# (display name, mcpat-report tag, m5out stats dir, architecture note)
# Add a row here if you introduce another architecture/workload combo --
# everything else in this script is generic over this list.
RUN_SPECS = [
    ("RVV / MinorCPU",     "rvv",          "m5out-rvv"),
    ("Scalar / MinorCPU",  "scalar",       "m5out-scalar"),
    ("RVV / O3CPU",        "rvv-o3",       "m5out-rvv-o3"),
    ("Scalar / O3CPU",     "scalar-o3",    "m5out-scalar-o3"),
    ("Scalar / U74",       "scalar-u74",   "m5out-scalar-u74"),
]


def parse_report(path):
    text = Path(path).read_text()
    out = {}
    for label, pattern in FIELDS:
        m = re.search(pattern, text)
        out[label] = float(m.group(1)) if m else None
    return out


def parse_sim_seconds(stats_path):
    if not stats_path.exists():
        return None
    text = stats_path.read_text()
    m = re.search(r"^simSeconds\s+([\d.eE+-]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def main():
    here = Path(__file__).resolve().parent.parent
    results = {}

    for display_name, tag, statsdir in RUN_SPECS:
        report_path = here / f"mcpat-{tag}-out.txt"
        if not report_path.exists():
            continue  # architecture/workload not run yet -- skip silently
        vals = parse_report(report_path)
        sim_s = parse_sim_seconds(here / statsdir / "stats.txt")
        vals["Sim Exec Time (s)"] = sim_s
        dynamic = vals.get("Runtime Dynamic (W)")
        vals["Energy / inference (mJ)"] = (
            dynamic * sim_s * 1000 if (dynamic is not None and sim_s is not None) else None
        )
        results[display_name] = vals

    if not results:
        sys.exit(
            "No mcpat-*-out.txt files found. Run scripts/run_mcpat.sh and/or "
            "scripts/run_mcpat_all.sh first."
        )

    columns = list(results.keys())
    col_w = max(22, max(len(c) for c in columns) + 2)

    header = f"{'Metric':<26}" + "".join(f"{c:>{col_w}}" for c in columns)
    print(header)
    print("-" * len(header))

    md_lines = ["| Metric | " + " | ".join(columns) + " |",
                "|---|" + "---|" * len(columns)]

    for label in ALL_LABELS:
        row_vals = [results[c].get(label) for c in columns]
        printed = [f"{v:.6f}" if v is not None else "n/a" for v in row_vals]
        print(f"{label:<26}" + "".join(f"{p:>{col_w}}" for p in printed))
        md_lines.append(f"| {label} | " + " | ".join(printed) + " |")

    missing_sim = [c for c in columns if results[c].get("Sim Exec Time (s)") is None]
    if missing_sim:
        note = (
            f"\nNote: simSeconds missing for: {', '.join(missing_sim)} -- "
            f"Sim Exec Time / Energy show n/a for those. Check "
            f"m5out-*/stats.txt exists and isn't empty for that run (a "
            f"gem5 run that exited before dumping stats produces this)."
        )
        print(note)
        md_lines.append(note)

    out_path = here / "comparison_all.md"
    out_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nWrote {out_path}")

    print(
        "\nReminder on how to read this across architectures (not just "
        "RVV vs scalar within one): MinorCPU and O3CPU columns are your "
        "own invented 22nm/100MHz core under two different gem5 timing "
        "models -- same STATIC ASSUMPTIONS, different pipeline behavior. "
        "U74 is a different core entirely (real SiFive silicon, fixed "
        "1.2GHz/32KiB), not a third guess at your target -- don't average "
        "it in with the other two or read it as 'more accurate,' just as "
        "a real-silicon reference point. See REFERENCES.md / the deep-"
        "review doc for what each column can and can't tell you."
    )


if __name__ == "__main__":
    main()
