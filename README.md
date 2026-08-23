# galmorph-rvs4 — `mcpat` branch (d108 architecture + full test/measurement suite)

This branch carries the **d108** S4D galaxy-morphology classifier (the improved,
higher-accuracy architecture) as a from-scratch, RVV-optimized C implementation,
**plus everything needed to reproduce all three tests**: functional accuracy,
RISC-V instruction count, and gem5 + McPAT power/energy.

## The model (d108)

RGB `3×64×64` image → 4×4 patchify + Hilbert scan (256 tokens × 48) →
linear up-projection (48→108) → **3 × [ S4D(d_model=108, 54 complex modes) →
GELU ]** → take-last → linear head (108→4) → softmax.

Optimizations carried over from the study: O(L·N) recurrent scan, `B̄`-into-`C̄`
fold, inline Remez math, and an RVV-vectorized scan (strip-mined because 54 modes
overflow one `m4` vector group). Validated at **82.20% accuracy, 100% agreement
with PyTorch** on the 2000-image GalaxyMNIST test set.

---

## Repository layout (this branch)

```
main.c                     the d108 model (single file; RVV + scalar paths)
weights.bin                d108 weights as a flat blob the C loader reads
weights.h / image.h        baked weights + a sample image, for the RISC-V build
Makefile                   make | make bench | make bench-scalar
profile.h                  per-layer instruction counter (RISC-V instret / x86 perf)
checkpoints/
  linear_d108_seed2_best.zip   the trained PyTorch checkpoint (source of weights)
scripts/
  make_d108_bins.py        checkpoint -> weights.bin / weights.h / image.h (no torch)
  kaggle_fresh_d108_test.py     accuracy test, self-contained Kaggle cell
  kaggle_accuracy_from_input.py accuracy test using an attached checkpoint dataset
  accuracy_d108.py         local C-vs-PyTorch accuracy over an exported test set
  export_testset_cell.py   notebook cell that exports the test set for accuracy_d108.py
  mcpat_full_dump.sh       one-shot dump of every gem5/McPAT/QEMU number, both archs
gem5-mcpat-pipeline/       gem5 + McPAT power/energy pipeline (build scripts, config,
                           converter, McPAT template)
docs/
  S4D_McPAT_Report.docx    the complete measured findings report
  S4D_McPAT_Results.xlsx   all raw numbers (both archs, RVV + scalar)
```

---

## Prerequisites (Ubuntu / WSL)

```bash
sudo apt update
sudo apt install -y build-essential git python3 python3-pip qemu-user
# RISC-V cross compiler with the V extension (skip if riscv32-unknown-elf-gcc is on PATH)
#   build riscv-gnu-toolchain --with-arch=rv32gcv --with-abi=ilp32f  (see gem5-mcpat-pipeline/)

# Python deps (numpy, torch, einops) go in their own environment, not system
# Python -- see "Step 0" in gem5-mcpat-pipeline/README.md for venv/conda setup:
python3 -m venv ~/.venvs/galmorph-rvs4 && source ~/.venvs/galmorph-rvs4/bin/activate
pip install -r requirements.txt

# gem5 + McPAT: one-time, see TEST 3 below -- deactivate the environment
# above first, gem5's build needs no pip packages and conda's bundled
# libraries can conflict with it.
```

---

## TEST 1 — RISC-V instruction count (QEMU)

Builds the baked RVV binary and runs one forward pass under QEMU, printing the
per-layer and total retired-instruction counts.

```bash
make bench CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench | sed -n '/Per-layer/,$p'
```

Scalar comparison:

```bash
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | grep TOTAL
```

Measured (one inference, −O2): **RVV ≈ 664M**, scalar ≈ 1.02B retired instructions.
(QEMU expands each cross-lane reduction into ~3100 instructions — this is the
optimization-study metric, not the gem5 committed count used for power.)

---

## TEST 2 — Functional accuracy vs PyTorch

The real galaxy test images live in the training environment (Kaggle), so the
accuracy test runs there. Two equivalent options:

**Option A — one self-contained Kaggle cell (recommended).** Add the checkpoint
`checkpoints/linear_d108_seed2_best.zip` as a Kaggle *input dataset*, enable
Internet, paste the contents of `scripts/kaggle_fresh_d108_test.py` into a cell,
and run. It downloads the test set, rebuilds the model, and prints:

```
PyTorch accuracy        : 82.20%
C-model (NumPy) accuracy: 82.20%
C vs PyTorch agreement  : 100.00%
```

The NumPy path in that cell is the exact math of `main.c` (verified equal to the
compiled C to 1.9e-10), so it stands in for the compiled binary.

**Option B — compiled binary over an exported test set.** In the notebook run
`scripts/export_testset_cell.py` (writes `test_d108/`), copy that folder next to
`build/main`, then locally:

```bash
make
python3 scripts/accuracy_d108.py     # runs ./build/main on each image, reports accuracy + agreement
```

Regenerate the weight files from the checkpoint at any time (no torch needed):

```bash
python3 scripts/make_d108_bins.py checkpoints/linear_d108_seed2_best.zip
```

---

## TEST 3 — Power & energy (gem5 + McPAT)

One-time build of gem5 and McPAT (see `gem5-mcpat-pipeline/` — run with **conda
deactivated**; gem5's build links against the apt packages from
`01_install_deps.sh`, and conda's bundled copies of those same libraries have
been found to conflict with it):

```bash
cd gem5-mcpat-pipeline
./01_install_deps.sh
./02_build_gem5.sh          # ~20–60 min
./03_build_mcpat.sh         # if it fails on -m32: sudo apt install gcc-multilib g++-multilib
```

Build both RISC-V binaries, then run the pipeline pointed at this repo:

```bash
# in the repo root:
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"

conda deactivate
cd gem5-mcpat-pipeline
REPO_DIR="$(cd .. && pwd)" ./scripts/run_both.sh     # runs RVV + scalar through gem5
./scripts/run_mcpat.sh                                # converts + runs McPAT
python3 scripts/compare_report.py                     # prints the Area/Power table
```

`compare_report.py` now prints and saves Sim Exec Time and Energy/inference
directly (Runtime-Dynamic power × `simSeconds` — gem5's simulated execution
time, not host wall-clock time, so these numbers match across machines).

**Get every number in one shot** (both architectures) with:

```bash
conda deactivate
bash scripts/mcpat_full_dump.sh
```

---

## Measured results (this machine — RiscvMinorCPU, 100 MHz, 22 nm McPAT, fp=0.60)

| Metric | d64 · RVV | d64 · Scalar | d108 · RVV | d108 · Scalar |
|---|---|---|---|---|
| gem5 committed instr | 33,082,055 | 152,657,262 | 19,948,328 | 33,051,024 |
| Exec time (sim s) | 0.382659 | 2.498141 | 0.286471 | 0.531889 |
| Runtime dynamic (W) | 0.010952 | 0.008756 | 0.009934 | 0.008931 |
| **Energy / inference** | **4.19 mJ** | 21.87 mJ | **2.85 mJ** | 4.75 mJ |

RVV advantage: **d64 5.22× less energy**, **d108 1.67× less energy**.
Fixed hardware (both): Area 1.567 mm², Peak Dynamic 0.0591 W, Total Leakage 0.00157 W.

Full details, per-component power breakdown, the QEMU-vs-gem5 counting explanation,
and caveats are in `docs/S4D_McPAT_Report.docx` and `docs/S4D_McPAT_Results.xlsx`.

---

## Notes

* Everything measured is **one forward pass** (one image); S4D compute is
  data-independent, so counts and energy are per-inference.
* The C model is validated: scalar path matches an independent NumPy reference to
  1.9e-10, and RVV matches scalar on-device.
* Absolute watts depend on the 22 nm / 100 MHz / fp-0.60 McPAT assumptions; the
  RVV-vs-scalar *ratios* are robust to them.
