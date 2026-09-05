# galmorph-rvs4

Energy and instruction efficiency of galaxy morphology classifiers on RISC-V,
written from scratch in C and hand optimized with the RISC-V Vector extension
(RVV 1.0). Correctness is checked against a NumPy reference, instruction counts
come from QEMU, and energy comes from gem5 plus McPAT.

This README is an ordered walkthrough. Follow it top to bottom. It is written so
that a machine with only gcc can start, and each new capability (instruction
counts, then energy) adds one small set of tools. A deeper reference with every
detail is in GUIDE.md.

Measurement basis for every energy number: RiscvMinorCPU, in order, 100 MHz,
VLEN 256, 22 nm McPAT, fp fraction 0.60, 16 KiB L1, one inference per run.

--------------------------------------------------------------------------------

## Quickstart (the whole thing)

```bash
# tools you need first (see step 1 for details and checks)
sudo apt install -y build-essential python3 python3-pip && pip3 install numpy

# get the code
git clone --branch dev https://github.com/Ahmed-Abdul-Rahim/galmorph-rvs4.git
cd galmorph-rvs4

# put the trained checkpoints into checkpoints/ (see step 3), then:
bash setup.sh                 # builds and validates every model, gcc only

# instruction counts and energy need more tools (steps 4 and 5), then:
bash scripts/run_all_benchmarks.sh
```

--------------------------------------------------------------------------------

## Step 1. Base tools: gcc and python

Install:
```bash
sudo apt update
sudo apt install -y build-essential python3 python3-pip
pip3 install numpy
```
Check (both must print a version):
```bash
gcc --version
python3 -c "import numpy; print('numpy', numpy.__version__)"
```

## Step 2. Get the code
```bash
git clone --branch dev https://github.com/Ahmed-Abdul-Rahim/galmorph-rvs4.git
cd galmorph-rvs4
```

## Step 3. Get the trained checkpoints

The weights are not committed. Put these six files into `checkpoints/`:
```
linear_d108_seed2_best.zip
d64_seq256__main__seed30485.pt
d108_seq256__main__seed30485.pt
d64_seq4096__main__seed30485.pt
d108_seq4096__main__seed30485.pt
cnn_only_large_61k__recipe_main__seed_8842__noise_floor_check_second_seed.pt.zip
```
The four seq length ones come together in the training export
`s4d_seqlen_test_weights.zip`:
```bash
unzip -j /path/to/s4d_seqlen_test_weights.zip "weights/*.pt" -d checkpoints/
```
Copy the d108 and CNN checkpoints in as well, then confirm:
```bash
bash scripts/fetch_checkpoints.sh
```

## Step 4. Build and validate every model (gcc only)
```bash
bash setup.sh
```
This bakes the weights from the checkpoints, builds each model with gcc, and
checks each one against an independent NumPy forward pass. Every model should
print `PASS` with a difference around 5e-05. If this works, the code is correct
on your machine and you are done with the correctness part.

To do one model by hand instead (example, the root d108):
```bash
python3 scripts/bake_weights.py --config d108 --ckpt-dir checkpoints
make
python3 scripts/validate_oracle.py --config d108 --bin ./build/main --ckpt-dir checkpoints
```

--------------------------------------------------------------------------------

## Step 5. Instruction counts (adds the RISC-V toolchain and QEMU)

Install QEMU:
```bash
sudo apt install -y qemu-user
qemu-riscv32 --version          # check
```
Install the RISC-V vector toolchain (skip if `which riscv32-unknown-elf-gcc`
already prints a path). Building it takes about an hour:
```bash
sudo apt install -y autoconf automake autotools-dev curl libmpc-dev libmpfr-dev \
                    libgmp-dev gawk bison flex texinfo gperf libtool patchutils \
                    bc libexpat-dev
git clone https://github.com/riscv-collab/riscv-gnu-toolchain ~/riscv-gnu-toolchain
cd ~/riscv-gnu-toolchain
./configure --prefix=/opt/riscv32 --with-arch=rv32gcv --with-abi=ilp32f
sudo make -j"$(nproc)"
echo 'export PATH="/opt/riscv32/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
riscv32-unknown-elf-gcc --version   # check
```
Run the counts for one model (example, d108_seq256):
```bash
cd variants/s4d-d108-seq256
python3 ../../scripts/bake_weights.py --config d108_seq256
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench        | sed -n '/Prediction/,$p'
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | grep TOTAL
```
Both builds must print the same Prediction line.

## Step 6. Energy (adds gem5 and McPAT)

Build gem5 and McPAT once (the gem5 build takes 20 to 60 minutes):
```bash
cd gem5-mcpat-pipeline
./01_install_deps.sh
./02_build_gem5.sh
./03_build_mcpat.sh
```
Check:
```bash
ls ~/gem5/build/RISCV/gem5.opt
ls ~/mcpat/mcpat
```
Run the energy for one model (conda off, gem5 needs system python):
```bash
conda deactivate
bash gem5-mcpat-pipeline/scripts/run_variant_energy.sh ~/galmorph-rvs4/variants/s4d-d108-seq256 d108_seq256
```
It prints two RESULT lines, RVV and scalar, each with committed, time, power,
energy.

## Step 7. Everything at once
```bash
conda deactivate
bash scripts/run_all_benchmarks.sh
```
Prints the QEMU counts for every S4D variant, then the energy commands to run.

--------------------------------------------------------------------------------

## The numbers you should see (RVV, MinorCPU, fp 0.60)

| model | committed | energy (mJ) | accuracy |
|-------|-----------|-------------|----------|
| d64_seq256  | 6,275,503  | 0.876 | 79.60% |
| d108_seq256 | 19,948,310 | 2.857 | 80.85% |
| d64_native  | 58,481,032 | 8.136 |        |

At a fixed 256 token sequence the larger model (d108) costs about 3.2 times the
energy of the smaller (d64), for 1.25 points of accuracy. That isolates model
width and depth from sequence length. Full tables and the RVV versus scalar
factors are in results/numbers.md and results/S4-Optimization.xlsx.

--------------------------------------------------------------------------------

## Repository map

```
main.c  Makefile  profile.h  pymodel/     Canonical S4D d108 and its Python reference
setup.sh                                    One command build and validate for every model
scripts/    bake_weights.py  validate_oracle.py  fetch_checkpoints.sh  run_all_benchmarks.sh
cnn/        galaxy-cnn/  mnist-cnn/          Convolutional models
variants/   s4d-d64-native  s4d-d64-seq256  s4d-d108-seq256   Other S4D builds
gem5-mcpat-pipeline/                         gem5 plus McPAT pipeline and energy runner
results/    S4-Optimization.xlsx  S4D_vs_CNN_Report.pdf  numbers.md
docs/       GUIDE.md (root)                  Methodology writeups and the full reference
checkpoints/                                 Trained weights, placed by you, not committed
```

Nothing generated is committed. Build outputs, baked headers, gem5 outputs, and
checkpoints are all produced by the steps above.

## Troubleshooting

- `weights.bin not found` when running a model: you skipped the bake step. Run
  `bash setup.sh`, or the three by hand commands in step 4.
- `Permission denied` on a pipeline script: `chmod +x gem5-mcpat-pipeline/scripts/*.sh gem5-mcpat-pipeline/*.sh`.
- gem5 will not start: run `conda deactivate` first; gem5 needs system python.
- More cases are in GUIDE.md section 8.

## Open item

The Galaxy CNN energy is not yet reproducible from this repo: its weight baking
script is not wired in. Everything on the S4D side reproduces exactly as listed.
