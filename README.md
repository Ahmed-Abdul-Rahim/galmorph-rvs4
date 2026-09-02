# galmorph-rvs4

Energy and instruction efficiency of galaxy morphology classifiers on RISC-V,
written from scratch in C and hand optimized with the RISC-V Vector extension
(RVV 1.0), measured with QEMU (instruction counts) and gem5 plus McPAT (energy).

Two model families are compared on the four class GalaxyMNIST task: a state space
model (S4D, in several sizes) and a small convolutional network (Galaxy CNN 61K).

## What is here

```
main.c  Makefile  profile.h  pymodel/     Canonical S4D d108 model and its Python reference
docs/                                       Methodology writeups and the McPAT report
scripts/                                    Weight baking, validation, accuracy, and the full suite
cnn/            galaxy-cnn/  mnist-cnn/     Convolutional models
variants/       s4d-d64-native  s4d-d64-seq256  s4d-d108-seq256   Other S4D builds
gem5-mcpat-pipeline/                        gem5 plus McPAT energy pipeline
results/        S4-Optimization.xlsx  S4D_vs_CNN_Report.pdf  numbers.md   All measured numbers
checkpoints/                                Trained weights (fetched, not committed)
```

Nothing generated is committed. Build outputs, baked headers (weights.h, image.h),
gem5 outputs, and checkpoints are all produced by the scripts. See INSTALL.md for
prerequisites and REPRODUCE.md for the exact commands that regenerate every number.

## Headline results (RiscvMinorCPU, VLEN 256, fp 0.60, one inference, RVV)

- Control experiment, both at 256 tokens: the larger S4D (d108, 3 layers) costs
  3.2 times the energy of the smaller (d64, 2 layers), for 1.25 points of accuracy.
  This isolates model size from sequence length.
- S4D d108 uses about 2.9 times less energy than the Galaxy CNN 61K per inference
  (2.846 mJ versus 8.228 mJ), despite the CNN having fewer parameters.

Full table in results/numbers.md and results/S4-Optimization.xlsx.

## Quick start

```bash
# 1. prerequisites (one time)
see INSTALL.md

# 2. checkpoints
bash scripts/fetch_checkpoints.sh

# 3. reproduce the numbers
see REPRODUCE.md      # instruction counts in minutes, energy in tens of minutes
```

## Validation

Every C model is checked against an independent NumPy forward pass built from the
same weights (scripts/validate_oracle.py). Agreement is about 4e-05, same predicted
class, so the compiled C matches the trained model math before any measurement.
