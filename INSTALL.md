# Install the prerequisites

Run everything on a WSL Ubuntu shell with sudo rights, in your own terminal.
Some steps take over an hour (the gem5 build and, if needed, the RISC-V toolchain
build). Nothing here is destructive.

## 1. System packages
```bash
sudo apt update
sudo apt install -y build-essential git python3 python3-pip python3-venv \
                    qemu-user gcc-multilib g++-multilib \
                    scons libprotobuf-dev protobuf-compiler libboost-all-dev \
                    zlib1g-dev m4 libpng-dev
```

## 2. Python
```bash
pip3 install numpy
```

## 3. RISC-V vector toolchain (riscv32-unknown-elf-gcc, V extension)
Check first:
```bash
which riscv32-unknown-elf-gcc
```
If it prints a path, skip ahead. If not, build it (about an hour):
```bash
sudo apt install -y autoconf automake autotools-dev curl libmpc-dev libmpfr-dev \
                    libgmp-dev gawk bison flex texinfo gperf libtool patchutils \
                    bc libexpat-dev
git clone https://github.com/riscv-collab/riscv-gnu-toolchain ~/riscv-gnu-toolchain
cd ~/riscv-gnu-toolchain
./configure --prefix=/opt/riscv32 --with-arch=rv32gcv --with-abi=ilp32f
sudo make -j"$(nproc)"
echo 'export PATH="/opt/riscv32/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 4. gem5 and McPAT
The pipeline ships build scripts. From the repo root:
```bash
cd gem5-mcpat-pipeline
./01_install_deps.sh
./02_build_gem5.sh          # builds build/RISCV/gem5.opt, 20 to 60 minutes
./03_build_mcpat.sh         # builds the mcpat binary
```
If step 3 fails on a 32 bit header error, the gcc-multilib package from section 1
fixes it. Rerun the script.

## 5. Verify everything resolves
```bash
which riscv32-unknown-elf-gcc qemu-riscv32
ls ~/gem5/build/RISCV/gem5.opt
ls ~/mcpat/mcpat
```
All four must print a real path.

## Notes
- gem5 needs system python, so run `conda deactivate` before any gem5 step.
- The McPAT converter reads the gem5 stat names used by this project's gem5. If a
  future gem5 renames stats, add the new names to the key lists in
  gem5-mcpat-pipeline/mcpat/gem5_to_mcpat.py.
