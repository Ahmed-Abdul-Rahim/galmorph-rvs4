#!/usr/bin/env bash
# gem5 stat names shift a little between versions and CPU models. Run this
# against your real stats.txt once, before your first McPAT conversion, and
# check the lines it finds against mcpat/gem5_to_mcpat.py's STAT_MAP dict.
#
# Usage: ./scripts/find_stats.sh m5out-rvv/stats.txt
set -euo pipefail
STATS="${1:?Usage: find_stats.sh <path-to-stats.txt>}"

section () { echo; echo "== $1 =="; }

section "cycles"
grep -iE "numCycles|\.cycles" "$STATS" | grep -i "^system.cpu" || echo "(no match)"

section "instructions committed"
grep -iE "numInsts|committedInsts|\.commit" "$STATS" | grep -i "^system.cpu" || echo "(no match)"

section "instruction type / class breakdown (int, float, branch, mem)"
grep -iE "InstType|op_class|IntAlu|FloatAdd|Branch|MemRead|MemWrite" "$STATS" || echo "(no match)"

section "branch predictor"
grep -iE "branchPred|mispredict|BTB" "$STATS" || echo "(no match)"

section "L1 instruction cache"
grep -iE "icache" "$STATS" | grep -iE "overallAccesses|overallMisses|ReadReq" || echo "(no match)"

section "L1 data cache"
grep -iE "dcache" "$STATS" | grep -iE "overallAccesses|overallMisses|ReadReq|WriteReq" || echo "(no match)"

section "TLB"
grep -iE "\.itb\.|\.dtb\.|tlb" "$STATS" | grep -iE "accesses|misses" || echo "(no match)"

section "simulated execution time -- USE THIS for energy/time comparisons across systems"
grep -E "^simSeconds|^simTicks|^sim_seconds|^sim_ticks" "$STATS" || echo "(no match)"

section "host wall-clock time -- do NOT use for energy/time comparisons (tracks THIS machine's speed running gem5, not the simulated core)"
grep -E "^hostSeconds|^hostTickRate|^host_seconds" "$STATS" || echo "(no match)"
