# GalMorph - A S4D-Based Galaxy Classifier Optimized for RVV Inference

A from-scratch C implementation of a diagonal Structured-State-Space (S4D)
galaxy-morphology classifier, optimized for inference on RISC-V Vector (RVV) 
processors

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Architecture](#2-model-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Dependencies](#4-dependencies)
5. [Usage](#5-usage)
6. [Benchmarking](#6-benchmarking)
7. [License](#7-license)
8. [References](#8-references)

---

## 1. Introduction

Spacecraft like JWST and Hubble capture astronomical imaging data faster
than it can ever be transmitted back: around 57 GB per day, with only hours
of downlink time. To be useful, a satellite needs to classify what it is
seeing, filter out the noise, and prioritize the observations worth
sending, all on-board, before anything reaches the ground.

The hardware doing that work is power-constrained and radiation-hardened.
There is no room for bloated runtimes. CNN inference scales with image
resolution and requires holding intermediate feature maps in memory, a
costly assumption on a space processor. Structured State Space models
process the same data as a sequence, with a fixed-size state that does not
grow with input length.

This repository implements S4D in C: a from-scratch forward pass with no
runtime dependencies beyond libc, targeting RISC-V Vector (RVV) extensions.
As a concrete workload, it classifies a 64x64 grayscale image into one of
four galaxy morphologies - Round Elliptical, In-between Elliptical,
Cigar-shaped Elliptical, Edge-on Disk - using GalaxyMNIST (see
[References](#8-references)) as a stand-in for the kind of on-board
imaging classification described above.

Everything needed to run it - math, layers, forward pass, and `main` -
lives in a single file: [`main.c`](main.c). The model is trained in
PyTorch (see the [`pymodel/`](pymodel/) package); this repository contains
the trained checkpoint and the from-scratch C reimplementation of its
forward pass, not the training pipeline.

## 2. Model Architecture

Galaxy classification is treated as a sequence modeling problem.

**Input processing:**
- Input: 64x64 RGB or grayscale images, $(B, C, 64, 64)$ where $C = 3$ for RGB or $C = 1$ for grayscale
- Hilbert scan reorders this into a sequence, $(B, 4096, C)$
- A linear input projection maps the $C$-dimensional pixels to the model dimension, $(B, 4096, d_{model})$

**S4D sequence processing:** two S4D layers are stacked, each with:
- State dimension $d_{state} = 64$ (memory capacity)
- Model dimension $d_{model} = 64$ (output feature dimension)
- GELU activation after each layer

**Classification head:**
- The final timestep of the sequence, $(B, 64)$, is taken as the summary representation
- A linear layer maps this to 4 class logits, $(B, 4)$
- Softmax converts the logits to class probabilities

**Forward pass:**

$$X_{img} \in \mathbb{R}^{C \times 64 \times 64} \xrightarrow{\text{Hilbert}} X_{seq} \in \mathbb{R}^{4096 \times C}$$

$$X_{seq} \xrightarrow{\text{Linear}} X_{proj} \in \mathbb{R}^{4096 \times 64}$$

$$X_{proj} \xrightarrow{\text{S4D}_1} Z_1 \in \mathbb{R}^{4096 \times 64} \xrightarrow{\text{GELU}} A_1$$

$$A_1 \xrightarrow{\text{S4D}_2} Z_2 \in \mathbb{R}^{4096 \times 64} \xrightarrow{\text{GELU}} A_2$$

$$A_2[:, -1, :] \in \mathbb{R}^{64} \xrightarrow{\text{Linear}} Y_{logits} \in \mathbb{R}^{4} \xrightarrow{\text{Softmax}} Y_{probs}$$

## 3. Repository Layout

```
.
├── pymodel                 # PyTorch model definition (exposes GalaxyClassifierS4D)
│   ├── __init__.py
│   ├── gclassifier.py
│   ├── hilbert.py
│   ├── s4d.py
│   └── tlts.py
├── scripts
│   └── generate_data.py    # regenerates weights.bin and the test data from model.pth
├── image.h                 # baked sample image, for the RISC-V benchmark build
├── main.c                  # the implementation: math, layers, forward pass, and main
├── Makefile                # builds the host binary or the baked RISC-V benchmark
├── model.pth               # the trained PyTorch checkpoint
├── profile.h               # per-layer instruction counter (RISC-V instret CSR, or x86 perf)
├── README.md
├── requirements.txt        # Python dependencies for scripts/generate_data.py
└── weights.h               # baked model weights, for the RISC-V benchmark build
```

## 4. Prerequisites - install everything

Commands below are for **Ubuntu / Debian / WSL**. On other systems install the same
tools with your package manager. Run them once; then jump to Section 5.

### 4.1 Build tools + Python
```bash
sudo apt update
sudo apt install -y build-essential git python3 python3-pip python3-venv
```

### 4.2 QEMU (RISC-V user-mode emulator, with the Vector extension)
```bash
sudo apt install -y qemu-user
qemu-riscv32 --version        # need >= 7.0 for RVV; newer is better
```
If your distro's QEMU is older than 7.0 (no vector support), build it from source:
<https://www.qemu.org/download/#source> (configure with `--target-list=riscv32-linux-user`).

### 4.3 RISC-V cross-compiler with the Vector extension
The RISC-V benchmark needs a `riscv32` toolchain that supports `rv32 + v` (vector).
**If `riscv32-unknown-elf-gcc` is already on your PATH, skip this.** Otherwise build it
once (~30-60 min):
```bash
sudo apt install -y autoconf automake autotools-dev curl python3 libmpc-dev \
     libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf \
     libtool patchutils bc zlib1g-dev libexpat-dev ninja-build cmake
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv32gcv --with-abi=ilp32f
sudo make                                    # builds riscv32-unknown-elf-gcc
echo 'export PATH=/opt/riscv/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
cd ..
riscv32-unknown-elf-gcc --version            # verify
```

### 4.4 (Optional) host instruction counter
Only for the *host* per-layer counts in Section 6. Not needed to run or validate.
```bash
sudo apt install -y linux-tools-common linux-tools-generic   # provides perf
```

---

## 5. Run it - copy-paste, top to bottom

### 5.1 Clone and generate the data
`weights.bin` and `test_data/` are **not** committed - they are generated from the
trained checkpoint `model.pth` (this step needs the Python packages):
```bash
git clone https://github.com/Ahmed-Abdul-Rahim/galmorph-rvs4.git
cd galmorph-rvs4
git switch c
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py          # writes weights.bin + test_data/*
```

### 5.2 Build and classify one image (host)
```bash
make
./build/main test_data/sample_0_img.bin   # -> Prediction: Round Elliptical (88.17%)
```

### 5.3 Validate against PyTorch - one command, 5/5
```bash
./build/main --validate
```

### 5.4 See all four classes
```bash
for c in round inbetween cigar edgeon; do
  printf "%-10s -> " "$c"; ./build/main test_data/variety_$c.bin | sed -n 's/^Prediction: //p'
done
```

### 5.5 Real RISC-V per-layer instruction counts (all optimization levels)
```bash
for O in 0 2 3; do
  make clean >/dev/null
  make bench CC=riscv32-unknown-elf-gcc CFLAGS="-O$O"
  echo "===== -O$O ====="
  qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench | sed -n '/Per-layer/,$p'
done
# least-of-three total ~= 1.85-2.1 billion instructions (~49x under the 91.2B baseline)
```

---

## 6. Benchmarking

Per-layer instruction counts are printed automatically by `main.c` via
`profile.h`. On host, they come from the Linux `perf` counter:

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1      # once per boot
mkdir -p build
gcc -O2 -o build/main main.c
./build/main test_data/sample_0_img.bin | sed -n '/Per-layer/,$p'
```

If the host PMU is not exposed, these read `0`; use the RISC-V path
instead, which reads a real `instret` counter inside QEMU:

```bash
make bench CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench | sed -n '/Per-layer/,$p'
```

`build/main` cannot run under QEMU (the newlib file-I/O limitation above),
which is why the RISC-V benchmark is a separate baked build (`build/bench`).

## 7. License

This project is licensed under the MIT License (license file to be added).

## 8. References

1. Gu, A., et al. (2022). *Efficiently Modeling Long Sequences with
   Structured State Spaces.* International Conference on Learning
   Representations (ICLR).
2. Gu, A., et al. (2022). *On the Parameterization and Initialization of
   Diagonal State Space Models.* Advances in Neural Information Processing
   Systems (NeurIPS).
3. Walmsley, M., et al. (2022). *Galaxy Zoo DECaLS: Detailed visual
   morphology measurements from volunteers and deep learning for 314,000
   galaxies.* Monthly Notices of the Royal Astronomical Society (MNRAS),
   509(3), 3966-3988.
4. Izard, A. (2022). *GalaxyMNIST: A Dataset for Galaxy Morphology
   Classification.* Available at: [albertizard.com/mnist](https://albertizard.com/mnist/)
5. Walmsley, M. (2022). *GalaxyMNIST Repository.* Available at:
   [github.com/mwalmsley/galaxy_mnist](https://github.com/mwalmsley/galaxy_mnist)
6. RISC-V International. *RISC-V Vector Extension Specification.*
   Available at: [github.com/riscv/riscv-v-spec](https://github.com/riscv/riscv-v-spec)
7. RISC-V International. *RISC-V in Space and Aerospace.* Available at:
   [riscv.org/risc-v-space-and-aerospace](https://riscv.org/risc-v-space-and-aerospace/)
