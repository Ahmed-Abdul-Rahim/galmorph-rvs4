# d108 architecture port (C implementation)

Ports the improved "linear_d108" model into the optimized C. Fully replaces the
old d64/2-layer model.

## What changed vs the previous C
| | old (d64) | new (d108) |
|---|---|---|
| input | grayscale 1x64x64 | **RGB 3x64x64** |
| tokenize | per-pixel, 4096 tokens | **4x4 patchify -> 256 tokens x 48** |
| up-projection | Linear(1->64) | **Linear(48->108)** |
| d_model | 64 | **108** |
| S4D modes | 32 complex | **54 complex** |
| S4D layers | 2 | **3** |
| head | Linear(64->4) | Linear(108->4) |

All optimizations carried over (recurrent scan, B_bar->C_bar fold, inline Remez
math, vectorized GELU). The one adaptation: 54 modes exceed one RVV `m4` group
(32 lanes @ VLEN=256), so the scan is **strip-mined** over the state each
timestep (state + constants in memory, ordered reduction chained across strips).

## Validation
The C forward pass was checked against an independent NumPy reference of the same
architecture, run on the real checkpoint weights: **max abs prob diff 1.9e-10,
same argmax.** (Scalar path validated on host; the RVV path is the same math
strip-mined ,  verify on-device by confirming the QEMU/gem5 prediction matches.)

## Files
- `main.c`         ported model (validated)
- `weights.bin`    the checkpoint as a flat blob the C loader reads
- `weights.h`,`image.h`  baked blobs for the RISC-V `make bench` build
- `make_d108_bins.py`  torch-free regenerator: checkpoint .zip -> the 3 files above
- `Makefile`       `make` / `make bench` / `make bench-scalar`
- `profile.h`      unchanged

## Build & run
```
make                                  # host build -> build/main
./build/main image.bin                # classify (weights.bin must be in cwd)
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"   # RVV baked build
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"   # scalar baked build
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench
```
Then feed build/bench and build/bench-scalar through the gem5+McPAT pipeline
exactly as before.

## Regenerate the bins from the checkpoint
```
python3 make_d108_bins.py linear_d108_seed2_best.zip   # -> weights.bin, weights.h, image.h
```

## Follow-up (not included)
`pymodel/` + `scripts/generate_data.py` still describe the d64 model. To produce
authoritative PyTorch `test_data/` for `--validate`, update those to the notebook's
d108 model (3 layers, patch 4, colored, s4_state=108). The C side is done.
