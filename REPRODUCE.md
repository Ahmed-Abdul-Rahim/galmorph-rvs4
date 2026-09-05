# Reproduce every number

Basis for every energy number: RiscvMinorCPU, 100 MHz, VLEN 256, 22 nm, fp 0.60,
16 KiB L1, one inference. Energy = runtime dynamic power times execution time.
Finish INSTALL.md first, and fetch the checkpoints:
```bash
bash scripts/fetch_checkpoints.sh
```

## The whole suite in one command
```bash
conda deactivate
bash scripts/run_all_benchmarks.sh
```
It prints the QEMU instruction counts for every S4D variant, then the exact gem5
plus McPAT commands to run for energy. The steps below show the same thing by hand.

## A. Bake weights for a model (from the fetched checkpoint)
```bash
cd variants/s4d-d108-seq256
python3 ../../scripts/bake_weights.py --config d108_seq256
```
Configs: d64_native, d64_seq256, d108_seq256. This writes weights.h and image.h,
which are not committed.

## B. Validate the C against the trained model
```bash
make                                   # host build that reads weights.bin
python3 ../../scripts/validate_oracle.py --config d108_seq256 --bin ./build/main
```
Expect a PASS with max difference about 4e-05.

## C. Instruction counts (QEMU, minutes)
```bash
make bench        CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
make bench-scalar CC=riscv32-unknown-elf-gcc CFLAGS="-O2"
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench        | sed -n '/Prediction/,$p'
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 ./build/bench-scalar | grep TOTAL
```
Both builds must print the same Prediction line.

## D. Energy (gem5 MinorCPU + McPAT)
```bash
conda deactivate
bash gem5-mcpat-pipeline/scripts/run_variant_energy.sh "$PWD" d108_seq256
```
Prints two RESULT lines, RVV and scalar, each with committed, time, power, energy.
Run one variant at a time (the pipeline reuses its output folders).

## Expected numbers
See results/numbers.md. The control experiment reproduces d64_seq256 at 0.876 mJ
and d108_seq256 at 2.857 mJ (RVV), a 3.2 times gap at equal sequence length.

## Accuracy
S4D accuracy runs on the GalaxyMNIST test set (Kaggle) with the cells in
scripts/ (kaggle_fresh_d108_test.py and friends). The CNN accuracy cell is still
to be added.
