# galmorph-rvs4 : the complete guide (install, build, test, reproduce)

This is the one document that takes a fresh machine to every measured number.
It installs every prerequisite, gets the code and the trained weights, builds the
tools, runs each model, and lists the exact numbers you should see.

Run everything on a WSL Ubuntu shell with sudo rights, in your own terminal.
Two steps are slow: building gem5 (20 to 60 minutes) and, only if you do not
already have it, building the RISC-V toolchain (about an hour). Everything else
is minutes.

Measurement basis for every energy number: RiscvMinorCPU, in order, 100 MHz,
VLEN 256, 22 nm McPAT, fp fraction 0.60, 16 KiB L1, one inference per run.
Energy = runtime dynamic power times execution time. Instruction counts are gem5
committed (retired) instructions. The RVV versus scalar and model to model ratios
are robust to the McPAT assumptions; the absolute watts depend on them.

--------------------------------------------------------------------------------

## 0. What you will end up with

Three ways to look at each model:
1. Correctness: the compiled C matches an independent NumPy forward pass built
   from the same trained weights, to about 4e-05.
2. Instruction counts: retired instructions under QEMU, RVV versus scalar.
3. Energy: runtime dynamic power and energy per inference from gem5 plus McPAT.

--------------------------------------------------------------------------------

## 1. System requirements

- WSL Ubuntu (or native Ubuntu) with sudo.
- About 6 GB free disk for gem5, McPAT, and the toolchain.
- Internet access for the one time installs.

--------------------------------------------------------------------------------

## 2. Install the prerequisites

Do these once. After each block there is a check so you know it worked.

### 2.1 System packages
```bash
sudo apt update
sudo apt install -y build-essential git python3 python3-pip python3-venv \
                    qemu-user gcc-multilib g++-multilib \
                    scons libprotobuf-dev protobuf-compiler libboost-all-dev \
                    zlib1g-dev m4 libpng-dev
```
Check:
```bash
qemu-riscv32 --version   # should print a version
gcc --version            # should print a version
```

### 2.2 Python
```bash
pip3 install numpy
```
Check:
```bash
python3 -c "import numpy; print('numpy', numpy.__version__)"
```

### 2.3 RISC-V vector toolchain (riscv32-unknown-elf-gcc, V extension)
First check whether you already have it:
```bash
which riscv32-unknown-elf-gcc
```
If it prints a path, skip to 2.4. If it prints nothing but you believe it is
installed, it is probably just not on PATH:
```bash
find / -maxdepth 6 -iname 'riscv32-unknown-elf-gcc*' 2>/dev/null
# if found under, say, /opt/riscv32/bin, add it:
echo 'export PATH="/opt/riscv32/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```
If you truly do not have it, build it (about an hour):
```bash
sudo apt install -y autoconf automake autotools-dev curl libmpc-dev libmpfr-dev \
                    libgmp-dev gawk bison flex texinfo gperf libtool patchutils \
                    bc libexpat-dev
git clone https://github.com/riscv-collab/riscv-gnu-toolchain ~/riscv-gnu-toolchain
cd ~/riscv-gnu-toolchain
./configure --prefix=/opt/riscv32 --with-arch=rv32gcv --with-abi=ilp32f
sudo make -j"$(nproc)"
echo 'export PATH="/opt/riscv32/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```
Check:
```bash
riscv32-unknown-elf-gcc --version
```
The `rv32gcv` architecture string is what enables the V (vector) extension.

### 2.4 gem5 and McPAT
The repository ships the build scripts. From the repo root (see section 3 to get
the repo first, then come back here):
```bash
cd gem5-mcpat-pipeline
./01_install_deps.sh          # scons, protobuf, boost, and friends
./02_build_gem5.sh            # builds build/RISCV/gem5.opt only, 20 to 60 minutes
./03_build_mcpat.sh           # builds the mcpat binary
```
If `03_build_mcpat.sh` fails on a 32 bit header error, the `gcc-multilib` package
from 2.1 fixes it; rerun the script. If it still fails, open McPAT's own Makefile
and remove the `-m32` flag from CXXFLAGS (McPAT does not need a 32 bit build of
itself).

Check:
```bash
ls ~/gem5/build/RISCV/gem5.opt
ls ~/mcpat/mcpat
```
Both must print a real path.

### 2.5 One shot verification of the whole toolchain
```bash
which riscv32-unknown-elf-gcc qemu-riscv32
ls ~/gem5/build/RISCV/gem5.opt
ls ~/mcpat/mcpat
```
All four lines must print a path. If any is empty, redo that part of section 2.

--------------------------------------------------------------------------------

## 3. Get the code and the trained weights

### 3.1 Clone (dev branch)
```bash
git clone --branch dev https://github.com/Ahmed-Abdul-Rahim/galmorph-rvs4.git
cd galmorph-rvs4
```

### 3.2 Checkpoints (kept out of git)
The trained weight files are not committed. Put them into `checkpoints/`. You
need these six:
```
linear_d108_seed2_best.zip
d64_seq256__main__seed30485.pt
d108_seq256__main__seed30485.pt
d64_seq4096__main__seed30485.pt
d108_seq4096__main__seed30485.pt
cnn_only_large_61k__recipe_main__seed_8842__noise_floor_check_second_seed.pt.zip
```
The four seq length checkpoints come together in the training export
`s4d_seqlen_test_weights.zip`:
```bash
unzip -j /path/to/s4d_seqlen_test_weights.zip "weights/*.pt" -d checkpoints/
```
Copy the d108 and CNN checkpoints in as well, then confirm:
```bash
bash scripts/fetch_checkpoints.sh
```
It prints one line per file and says whether each is present.

--------------------------------------------------------------------------------

## 4. The models

| where | model | d_model | state | layers | tokens | params | role |
|-------|-------|---------|-------|--------|--------|--------|------|
| root  | S4D d108 | 108 | 108 | 3 | 256 | 76,360 | canonical S4D |
| variants/s4d-d108-seq256 | S4D d108 | 108 | 108 | 3 | 256 | 76,360 | control, large |
| variants/s4d-d64-seq256  | S4D d64  | 64  | 64  | 2 | 256 | 20,036 | control, small |
| variants/s4d-d64-native  | S4D d64  | 64  | 64  | 2 | 4096 | 17,156 | native long sequence |
| cnn/galaxy-cnn | Galaxy CNN 61K | conv stem | | | 256 patches | 61,044 | CNN comparison |
| cnn/mnist-cnn  | MNIST CNN | conv | | | | | reference CNN |

The four galaxy classes are Round Elliptical, In-between Elliptical, Cigar-shaped
Elliptical, and Edge-on Disk.

--------------------------------------------------------------------------------

## 5. Test one model, step by step

The example uses `variants/s4d-d108-seq256`. The same four steps work for every
S4D variant (swap the folder and the `--config` name: d108_seq256, d64_seq256,
d64_native).

### 5.1 Bake the weights from the checkpoint
```bash
cd variants/s4d-d108-seq256
python3 ../../scripts/bake_weights.py --config d108_seq256
```
Writes `weights.h` and `image.h` (both generated, both gitignored).

### 5.2 Confirm the C matches the trained model
```bash
make                                                     # host build, reads weights.bin
python3 ../../scripts/validate_oracle.py --config d108_seq256 --bin ./build/main
```
Expect: `PASS` with max difference about 4e-05, same predicted class.

### 5.3 Instruction counts (QEMU, minutes)
```bash
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench        | sed -n '/Prediction/,$p'
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | grep TOTAL
```
Both builds must print the same Prediction line. That is the on device proof that
the vector path is numerically correct.

### 5.4 Energy (gem5 MinorCPU plus McPAT)
```bash
conda deactivate     # gem5 needs system python
bash gem5-mcpat-pipeline/scripts/run_variant_energy.sh "$PWD" d108_seq256
```
Prints two RESULT lines, RVV and scalar, each with committed, time, power, energy.
Run one variant at a time (the pipeline reuses its output folders).

--------------------------------------------------------------------------------

## 6. Run the whole suite at once
```bash
conda deactivate
bash scripts/run_all_benchmarks.sh
```
It bakes, builds, and prints the QEMU instruction counts for all three S4D
variants, then prints the exact energy commands to run.

--------------------------------------------------------------------------------

## 7. The numbers you should see

All reproduced from a clean checkout with the repository McPAT converter.

### 7.1 Energy (gem5 MinorCPU, fp 0.60), per inference
| model | build | committed | time (s) | power (W) | energy (mJ) |
|-------|-------|-----------|----------|-----------|-------------|
| d64_seq256  | RVV    | 6,275,503   | 0.0864 | 0.010135 | 0.876  |
| d64_seq256  | Scalar | 12,468,631  | 0.2013 | 0.008681 | 1.748  |
| d108_seq256 | RVV    | 19,948,310  | 0.2880 | 0.009920 | 2.857  |
| d108_seq256 | Scalar | 33,053,634  | 0.5320 | 0.008974 | 4.774  |
| d64_native  | RVV    | 58,481,032  | 0.7156 | 0.011370 | 8.136  |
| d64_native  | Scalar | 157,853,559 | 2.5496 | 0.008701 | 22.185 |

### 7.2 Instruction counts (QEMU retired, minus O2)
| model | RVV | Scalar |
|-------|-----|--------|
| d64_seq256  | 182,937,262   | 320,704,719   |
| d108_seq256 | 670,119,168   | 1,047,593,863 |
| d64_native  | 2,288,293,276 | 4,597,823,450 |
(QEMU retired counts differ from gem5 committed because QEMU expands each cross
lane vector reduction into thousands of retired instructions; gem5 counts it as
one. The energy uses the gem5 committed count.)

### 7.3 The two headline results
- Control experiment, both at 256 tokens: the larger model (d108) costs 3.18
  times the instructions and 3.26 times the energy of the smaller (d64), for
  1.25 points of accuracy. This isolates model width and depth from sequence
  length; the earlier apparent advantage of d108 over native d64 was the shorter
  sequence, not the width.
- RVV versus scalar energy: d64_seq256 2.00x, d108_seq256 1.67x, d64_native 2.73x.

### 7.4 Accuracy (GalaxyMNIST test set)
- S4D d108 (linear_d108_seed2): 82.2 percent, C matches PyTorch to the fp32 floor.
- Seed 30485 runs: d64_seq256 79.60, d108_seq256 80.85 percent.
- Galaxy CNN 61K: not yet measured on the test set.

### 7.5 Predicted class on the baked sample image (deterministic)
- d108_seq256: Edge-on Disk. d64_seq256: Round Elliptical. d64_native: Round
  Elliptical. RVV and scalar must agree per model.

Full history and the earlier VLEN 512 study are in
results/S4-Optimization.xlsx. A short write up is results/S4D_vs_CNN_Report.pdf.

--------------------------------------------------------------------------------

## 8. Troubleshooting

- `Permission denied` on a pipeline script: it lost its execute bit. Fix with
  `chmod +x gem5-mcpat-pipeline/scripts/*.sh gem5-mcpat-pipeline/*.sh`.
- gem5 fails to start or cannot find python: run `conda deactivate` first. gem5
  needs system python.
- McPAT converter says a required stat like `total_cycles` is missing: your gem5
  uses different stat names. Add them to the key lists in
  `gem5-mcpat-pipeline/mcpat/gem5_to_mcpat.py`. This project's gem5 uses
  `board.processor.cores.core.numCycles` and
  `board.processor.cores.core.commitStats0.committedInstType::total`.
- McPAT build fails on `-m32`: install `gcc-multilib g++-multilib`, or remove the
  `-m32` flag from McPAT's Makefile.
- `checkpoint not found`: run `bash scripts/fetch_checkpoints.sh` and put the
  named files into `checkpoints/`.
- Energy runner reports empty power: the McPAT run produced no output. Confirm
  `~/mcpat/mcpat` exists and that the variant was built (`make bench` first).

--------------------------------------------------------------------------------

## 9. Repository map

```
main.c  Makefile  profile.h  pymodel/     Canonical S4D d108 and its Python reference
docs/                                       Methodology writeups and the McPAT report
scripts/     bake_weights.py  validate_oracle.py  fetch_checkpoints.sh
             run_all_benchmarks.sh  make_d108_bins.py  accuracy and kaggle cells
cnn/         galaxy-cnn/  mnist-cnn/         Convolutional models
variants/    s4d-d64-native  s4d-d64-seq256  s4d-d108-seq256   Other S4D builds
gem5-mcpat-pipeline/                         gem5 plus McPAT pipeline and energy runner
results/     S4-Optimization.xlsx  S4D_vs_CNN_Report.pdf  numbers.md
checkpoints/                                 Trained weights, fetched, not committed
```

Nothing generated is committed. Build outputs, baked headers, gem5 outputs, and
checkpoints are all produced by the steps above.

--------------------------------------------------------------------------------

## 10. Known open item

The Galaxy CNN energy (about 8.2 mJ RVV) was measured with a different McPAT
converter and its weight baking script is not yet wired into this repo. Until it
is, the CNN number is not reproducible end to end here, and it should be
re-measured with the repository converter so it lines up with the S4D numbers.
Everything on the S4D side reproduces exactly as listed in section 7.
