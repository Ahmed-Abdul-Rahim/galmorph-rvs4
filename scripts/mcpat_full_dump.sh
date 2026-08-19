#!/usr/bin/env bash
# Comprehensive McPAT/gem5/QEMU dump for d64 + d108.  Run with conda OFF.
#   conda deactivate ; bash ~/mcpat_full_dump.sh
unset PYTHONHOME PYTHONPATH
PIPE="$HOME/gem5-mcpat-pipeline"
QEMU="qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32"

dump () {
  echo "########## $1 ##########"
  for tag in rvv scalar; do
    s="$PIPE/m5out-$tag/stats.txt"; m="$PIPE/mcpat-$tag-out.txt"
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
}

echo "==== QEMU instruction totals (one forward pass, -O2 baked binaries) ===="
echo "d64  RVV    $($QEMU $HOME/galmorph-rvs4/build/bench        2>/dev/null | awk '/TOTAL/{print $3}')"
echo "d64  scalar $($QEMU $HOME/galmorph-rvs4/build/bench-scalar 2>/dev/null | awk '/TOTAL/{print $3}')"
echo
dump "D64  (galmorph-rvs4)  -- current gem5/McPAT outputs"

echo; echo ">>> building + running D108 through gem5/McPAT (silent, ~3-5 min)..."
cd "$HOME/galmorph-d108" && make clean >/dev/null 2>&1
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2" >/dev/null 2>&1
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2" >/dev/null 2>&1
echo "d108 RVV    $($QEMU $HOME/galmorph-d108/build/bench        2>/dev/null | awk '/TOTAL/{print $3}')"
echo "d108 scalar $($QEMU $HOME/galmorph-d108/build/bench-scalar 2>/dev/null | awk '/TOTAL/{print $3}')"
cd "$PIPE" && REPO_DIR="$HOME/galmorph-d108" ./scripts/run_both.sh >/dev/null 2>&1
./scripts/run_mcpat.sh >/dev/null 2>&1
echo
dump "D108 (galmorph-d108)"
echo; echo "==== DONE — select all this output and paste it back ===="
