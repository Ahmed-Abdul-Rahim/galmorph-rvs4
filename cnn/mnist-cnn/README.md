# RVV-optimized MNIST CNN (from the TA's riscv-assembly-cnn, S4D-style)

Single-file, RVV-vectorized version of the TA's C CNN:
`28x28 → Conv(5x5,8) → ReLU → MaxPool(2x2) → flatten(1152) → Dense(1152→10) → softmax → argmax`

## What was optimized (RVV 1.0 intrinsics inline, scalar fallback)
- **conv2d**: the commented-out **im2col + GEMM** path, vectorized **across patches**
  (rank-1 `vfmacc` accumulation, accumulator kept resident) ,  no cross-lane reductions.
- **ReLU**: vectorized `vfmax`.
- **dense**: the S4D-style `linear()` ,  vectorized **across the 10 outputs**
  (`vfmacc` into a 10-lane accumulator), contiguous weight loads.
- **softmax**: Taha's **range-reduced Horner exp** (`range_reduced_exp_horner`) replaces
  the divergent `exp_taylor(x,1000)`, **plus max-subtraction** for numerical stability.
- maxPool / flatten kept scalar (cheap, not the bottleneck).

## Validation
Every stage (conv, relu, maxpool, flatten, dense) matches the TA's baseline C **exactly
(0.0 diff)** on host; prediction = **6** (same as baseline). The softmax is now a proper
distribution (0.99882 @ 6) ,  the baseline's Taylor softmax was numerically broken
(values >1, negatives).

## Build & run
```bash
make                 # host build -> build/cnn  (scalar, for validation)
./build/cnn

# RISC-V instruction counts under QEMU (RVV vs scalar), like S4D:
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench        | sed -n '/Per-stage/,$p'
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | sed -n '/Per-stage/,$p'
```
The RVV and scalar builds should print the **same prediction (6)** ,  that confirms the
vector path is correct on-device ,  and the per-stage instruction counts give the speedup.

## Files
`cnn_opt.c` (the model) · `config.h` (weights, from the TA repo) · `image_data.h` (baked
28×28 image) · `profile.h` (instret counter) · `Makefile`.

## Not included (per discussion)
Group norm ,  this model has none; add later if a newer model needs it.
