# RVV-optimized Galaxy CNN — GalaxyClassifierGrid, Stem4Full (4L) + 0 S4D

Single-file, RVV-vectorized C implementation of the 38,468-param galaxy
classifier (the "stem_full (4L) + 0 S4D" model), for the same gem5/McPAT
comparison as the S4D work.

## Architecture (inference)
```
3x64x64
Conv(3->32,3x3,s1,p1) -> GroupNorm(8,32) -> GELU              32x64x64
res: Conv(32->32,s1,p1) -> GN(8,32) -> GELU ;  x = x + res    32x64x64
Conv(32->32,3x3,s2,p1) -> GN(8,32) -> GELU                    32x32x32
Conv(32->64,3x3,s2,p1) -> GN(8,64) -> GELU                    64x16x16
GlobalAvgPool -> Linear(64->4) -> softmax
```

## Optimizations (RVV 1.0 intrinsics inline, scalar fallback)
- **conv**: im2col + GEMM, vectorized ACROSS patches (no cross-lane reductions),
  generic over channels / stride / padding — reuses the TA's im2col+GEMM idea.
- **GroupNorm(8)**: per-group mean/var then affine normalize.
- **GELU**: tanh-approx with inline Remez exp/tanh (vectorized), same math as S4D.
- **Linear(64->4)**: S4D-style, vectorized across the outputs.
- **softmax**: range-reduced Horner exp (Taha's hpp) + max-subtraction.

## Weights
Deterministic placeholder weights are generated at startup (before the timed
region), because instruction count / cycles / energy are **weight-independent**.
Drop in the trained `Stem4Full` checkpoint later for accuracy — the compute
numbers don't change.

## Build & measure (same as S4D/CNN)
```bash
make                                              # host build (scalar) -> build/galaxy
./build/galaxy

make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench        | sed -n '/Per-stage/,$p'
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | sed -n '/Per-stage/,$p'
```
Both should print the same predicted class (confirms the vector path). Then feed
`build/bench` / `build/bench-scalar` through the gem5+McPAT pipeline exactly like
S4D (point REPO_DIR at this folder) to get energy per inference.

## Files
`galaxy_cnn.c` · `profile.h` · `Makefile`
