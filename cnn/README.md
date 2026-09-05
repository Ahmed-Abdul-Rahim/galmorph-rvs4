# CNN branch ,  RVV-optimized CNNs (companion to the S4D work)

Two from-scratch, RVV-vectorized C CNNs, same methodology as the S4D models
(inline RVV intrinsics + scalar fallback, im2col+GEMM conv, range-reduced
Horner exp), for gem5/McPAT comparison.

## galaxy-cnn/  ,  GalaxyClassifierGrid, Stem4Full (4L) + 0 S4D  (38,468 params)
`3x64x64 → [Conv→GroupNorm(8)→GELU] x4 (+residual) → GlobalAvgPool → Linear(64→4) → softmax`
Measured (1 inference, VLEN=256, QEMU): **RVV 953.6M vs Scalar 5,103.6M instructions = 5.35x fewer.**
RVV and scalar agree on the prediction (vector path verified on-device).
Weights are deterministic placeholders (compute is weight-independent; drop in the
trained Stem4Full checkpoint for accuracy).

## mnist-cnn/  ,  the TA's riscv-assembly-cnn, optimized
`28x28 → Conv(5x5,8) → ReLU → MaxPool → Dense(1152→10) → softmax`
Every stage matches the TA's baseline exactly; softmax fixed (Horner exp + max-subtract).

## Build & measure (each folder)
```bash
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench | sed -n '/Per-stage/,$p'
```
Then feed build/bench + build/bench-scalar through the gem5+McPAT pipeline
(REPO_DIR = that folder) for energy, exactly like S4D.

## Headline comparison (gem5+McPAT, RVV, one inference)
| model | energy/inf | committed instr |
|---|---|---|
| S4D d108           | 2.85 mJ | 19.9M |
| Galaxy CNN (4L)    | 6.05 mJ | 40.2M |
