# S4D model variants

Self-contained builds of the S4D classifier used in the study. Each folder has
its own `main.c`, `profile.h`, and `Makefile`. The baked headers `weights.h`
and `image.h` are generated, not committed. Generate them first with the weight
baking script, then build.

| folder | d_model | state | layers | tokens (seq_len) | role |
|--------|---------|-------|--------|------------------|------|
| s4d-d64-native   | 64  | 64  | 2 | 4096 (per pixel) | original d64, native long sequence |
| s4d-d64-seq256   | 64  | 64  | 2 | 256  (4x4 patch) | control experiment, small model |
| s4d-d108-seq256  | 108 | 108 | 3 | 256  (4x4 patch) | control experiment, large model |

The canonical d108 model at the repository root is the same architecture as
s4d-d108-seq256; it lives at the root because the docs and Python reference are
written against it.

## Build one variant
```bash
cd variants/s4d-d64-seq256
python3 ../../scripts/bake_weights.py --config d64_seq256   # writes weights.h, image.h
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench | sed -n '/Prediction/,$p'
```
Energy: point the pipeline runner at the folder (see REPRODUCE.md).
