#!/usr/bin/env bash
# Complete pipeline script: Builds, simulates via gem5, runs McPAT, and dumps stats.

unset PYTHONHOME PYTHONPATH
REPO_DIR="$(pwd)"
PIPE="$REPO_DIR/gem5-mcpat-pipeline"
QEMU="qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32"

echo "==== 1. QEMU Instruction Totals (One Forward Pass) ===="
echo "RVV    $($QEMU "$REPO_DIR/build/bench"        2>/dev/null | awk '/TOTAL/{print $3}')"
echo "scalar $($QEMU "$REPO_DIR/build/bench-scalar" 2>/dev/null | awk '/TOTAL/{print $3}')"

echo ""
echo "==== 2. Running gem5 Simulation & McPAT Analysis (~3-5 mins) ===="
cd "$PIPE" || exit 1
REPO_DIR="$REPO_DIR" ./scripts/run_both.sh >/dev/null 2>&1
./scripts/run_mcpat.sh >/dev/null 2>&1
cd "$REPO_DIR" || exit 1

echo ""
echo "########## gem5 / McPAT Summary Output ##########"
for tag in rvv scalar; do
  s="$PIPE/m5out-$tag/stats.txt"
  m="$PIPE/mcpat-$tag-out.txt"
  echo "----- $tag -----"
  echo "simSeconds  $(grep -m1 '^simSeconds' "$s" | awk '{print $2}')"
  echo "numCycles   $(grep -m1 'cores.core.numCycles' "$s" | awk '{print $2}')"
  echo "committed   $(grep -m1 'committedInstType::total' "$s" | awk '{print $2}')"
  echo "ProcArea    $(grep -m1 '^  Area =' "$m" | awk '{print $3}')"
  echo "PeakDynamic $(grep -m1 '^  Peak Dynamic =' "$m" | awk '{print $4}')"
  echo "SubLeak     $(grep -m1 '^  Subthreshold Leakage =' "$m" | awk '{print $4}')"
  echo "GateLeak    $(grep -m1 '^  Gate Leakage =' "$m" | awk '{print $4}')"
  echo "RuntimeDyn  $(grep -m1 '^  Runtime Dynamic =' "$m" | awk '{print $4}')"
  echo "ExecUnit    $(awk '/Execution Unit:/{f=1} f&&/Runtime Dynamic =/{print $4; exit}' "$m")"
  awk '/Execution Unit:/{e=1}
       e&&/Integer ALUs/{ia=1} ia&&/Runtime Dynamic =/{print "  IntALU   "$4; ia=0}
       e&&/Floating Point Units/{fp=1} fp&&/Runtime Dynamic =/{print "  FPU      "$4; fp=0}
       e&&/Complex ALUs/{cx=1} cx&&/Runtime Dynamic =/{print "  ComplexA "$4; cx=0}
       /Results Broadcast/{e=0}' "$m"
done

echo ""
echo "==== DONE ===="