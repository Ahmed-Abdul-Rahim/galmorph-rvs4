# S4D Galaxy Classifier — Optimized C (single file)

A from-scratch C implementation of a diagonal Structured-State-Space (**S4D**)
galaxy-morphology classifier, optimized down to **1.82 billion** dynamic instructions —
**50× fewer** than the original convolution baseline — with **no libm** in the forward
pass and **identical predictions**.

It classifies a 64×64 grayscale image into one of four classes
(**Round Elliptical, In-between Elliptical, Cigar-shaped Elliptical, Edge-on Disk**):

```
image → Hilbert scan → linear up-projection → S4D → GELU → S4D → GELU
      → take-last-timestep → linear head → softmax
```

Everything is in **one file, `galaxy_s4d.c`** — math, layers, forward pass, and `main`.
The full optimization story is in [`docs/`](docs/00-overview.md).

---

## 1. What's in this branch

```
galaxy_s4d.c            THE implementation — math + all layers + forward pass + main
profile.h               per-layer instruction counter (RISC-V instret CSR, or x86 perf)
Makefile                one-command build (host or RISC-V)
model_params/
  model_weights.bin     the flat weight blob the program loads (21124 floats)
  galaxys4-30609.pth    the trained PyTorch checkpoint (reference / to regenerate weights)
test_data/              per-sample inputs + PyTorch reference outputs (for validation)
docs/                   one markdown per optimization, fully explained
```

---

## 2. Quick start — host (x86), ~5 seconds

Run from the **repository root** (it reads `model_params/model_weights.bin`):

```bash
make
./galaxy_app test_data/sample_0_img.bin      # -> Prediction: Round Elliptical (88.17%)
```

It prints the prediction and a per-layer instruction-count table. Those host counts use
the Linux `perf` counter, which is off by default — enable it once, then you can see the
breakdown across optimization levels:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1      # once per boot
for O in 0 2 3; do
  gcc -O$O -o galaxy_app galaxy_s4d.c
  echo "===== -O$O ====="; ./galaxy_app test_data/sample_0_img.bin | sed -n '/Per-layer/,$p'
done
```

These are **real x86** retired-instruction counts. Note: inside a VM/WSL the host PMU
is often not exposed, so these may read `0` even with the sysctl set — in that case use
the RISC-V measurement in §3, which reads a real counter inside QEMU. Either way, the
RISC-V numbers (§3) are the ones the study reports.

---

## 3. Real RISC-V instruction counts (the study's numbers)

`qemu-riscv32`'s newlib C library **cannot open files**, so the RISC-V measurement uses a
**baked** build (`-DBAKED`) with the weights + a sample image compiled in via
`bench_data.h`. Your toolchain's default arch already includes the vector extension, so
do **not** pass `-march` (`rv32gcv` has no multilib on the standard toolchain):

```bash
for O in 0 2 3; do
  make clean >/dev/null
  make bench CC=riscv32-unknown-elf-gcc CFLAGS="-O$O"
  echo "===== -O$O ====="
  qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./galaxy_bench | sed -n '/Per-layer/,$p'
done
```

`galaxy_bench` takes **no arguments** (the image is baked in). The counts come straight
from the `instret` CSR — exact and inlining-proof — and the headline figure is the
**least of the three** builds (~**1.82 B** total). Run at `vlen=256` (matches the VeeR
config); the QEMU default of 128 would process half of each 32-lane group and be wrong.

> The file-loading `galaxy_app` is for the host only — it will *not* run under QEMU
> because of the newlib file-I/O limitation above. That is why the benchmark is baked.

---

## 4. Validate correctness — one command, no python

The program has a built-in validation mode: it runs samples 0-4 and compares its softmax
to the PyTorch reference (`test_data/sample_N_softmax.bin`).

```bash
make
./galaxy_app --validate
```

Expected:

```
Validating against PyTorch reference (test_data/):
  sample 0: Round Elliptical       vs ref Round Elliptical       | max prob diff 2.62e-05 | MATCH
  sample 1: Edge-on Disk           vs ref Edge-on Disk           | max prob diff 9.34e-05 | MATCH
  sample 2: Edge-on Disk           vs ref Edge-on Disk           | max prob diff 1.06e-09 | MATCH
  sample 3: Edge-on Disk           vs ref Edge-on Disk           | max prob diff 2.33e-06 | MATCH
  sample 4: Edge-on Disk           vs ref Edge-on Disk           | max prob diff 2.38e-07 | MATCH

5/5 classes match the PyTorch reference  (probabilities match to the fp32 floor)
```

The `max prob diff ~1e-5` is the fp32 precision floor of the from-scratch polynomial math;
the **classification is identical to PyTorch** on every sample.

### Try all four classes

`test_data/` also has one synthetic input per class (random inputs chosen to exercise
each output), so you can watch the classifier hit every label:

```bash
for c in round inbetween cigar edgeon; do
  printf "%-10s -> " "$c"; ./galaxy_app test_data/variety_$c.bin | sed -n 's/^Prediction: //p'
done
```

Expected: Round Elliptical, In-between Elliptical, Cigar-shaped Elliptical, Edge-on Disk.
(These are synthetic — noise inputs, no PyTorch reference — purely to demonstrate all four
outputs; the numbered `sample_N` files are the real PyTorch-validated cases.)

## 5. Weight format (`model_params/model_weights.bin`)

A single flat little-endian blob, read in this order (see `model_forward` in
`galaxy_s4d.c`):

```
hilbert_scan.indices   4096 × int32
uproject.weight        64 × 1  float32
uproject.bias          64      float32
s4_1: log_dt(64), log_A_real(64×32), A_imag(64×32), C(64×32×2 interleaved re/im), D(64)
s4_2: same shape as s4_1
fc.weight              4 × 64  float32
fc.bias                4       float32
```

Total = 21124 float-sized words. To regenerate it (and the reference tensors) from the
checkpoint `model_params/galaxys4-30609.pth`, use the export script on the Python branch
of this project.

---

## 6. The optimization story

| # | Change | Instructions | Doc |
|--:|---|--:|---|
| — | Original: O(L²) conv + Taylor math | 91.22 B | — |
| 1 | recurrent O(L·N) scan | 6.87 B | [01](docs/01-recurrent-scan.md) |
| 2 | Remez minimax math | 5.74 B | [02](docs/02-remez-math.md) |
| 3–5 | scan refinements | 3.09 B | [03](docs/03-scan-refinements.md) |
| 8 | RVV vectorized scan | 2.68 B | [04](docs/04-rvv-vectorization.md) |
| 12 | fold B̄ into C̄ (−30%) | 1.87 B | [05](docs/05-fold-b-into-c.md) |
| 13–14 | inline math + vectorized GELU | **1.82 B** | [06](docs/06-inline-math-and-gelu.md) |

Things that were tried and **didn't** help are documented too —
[`docs/07-negative-results.md`](docs/07-negative-results.md) — including why the
reduction couldn't be made cheaper under QEMU.

---

## 7. Requirements

- **Host build & validation:** any C99 compiler (`gcc`/`clang`). Nothing else — the
  `--validate` check is built in (no python/numpy). Host per-layer counts additionally
  need Linux `perf` access (often unavailable in a VM/WSL; use the RISC-V path instead).
- **RISC-V build:** `riscv32-unknown-elf-gcc` with `rv32gcv` / `ilp32f`, and
  `qemu-riscv32` (or VeeR-iSS) run at **`vlen=256`**.
- No third-party libraries; the math is all in `galaxy_s4d.c`.
