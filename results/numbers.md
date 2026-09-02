# Measured numbers (for the paper)

Basis for every row: RiscvMinorCPU, in order, 100 MHz, VLEN 256, 22 nm McPAT,
fp fraction 0.60, 16 KiB L1, one inference. Energy = runtime dynamic power times
execution time. Instruction counts are gem5 committed (retired). All C models are
validated against an independent NumPy forward pass to about 4e-05.

## Controlled experiment: fixed sequence length (256 tokens), RVV
| model | d_model | layers | committed | energy (mJ) | accuracy |
|-------|---------|--------|-----------|-------------|----------|
| d64_seq256  | 64  | 2 | 6,268,591  | 0.931 | 79.60% |
| d108_seq256 | 108 | 3 | 19,948,310 | 3.010 | 80.85% |

At a fixed 256 token sequence the larger model costs 3.2 times the instructions
and 3.2 times the energy, for 1.25 points of accuracy. Isolates width and depth
from sequence length.

## CNN versus S4D (RVV)
| model | committed | energy (mJ) |
|-------|-----------|-------------|
| Galaxy CNN 61K | 53,825,016 | 8.228 |
| S4D d108        | 19,948,328 | 2.846 |

S4D d108 uses about 2.9 times less energy than the CNN per inference. Parameter
counts: CNN 61,044; S4D d108 76,360. The CNN is heavier despite fewer parameters
because im2col plus GEMM does more arithmetic per parameter than the S4D scan.

## RVV versus scalar (fixed sequence 256)
| model | fewer instr | faster | less energy |
|-------|-------------|--------|-------------|
| d64_seq256  | 1.99x | 2.42x | 2.07x |
| d108_seq256 | 1.66x | 1.86x | 1.70x |

## Accuracy (GalaxyMNIST test set)
- S4D d108 (linear_d108_seed2): 82.2 percent, C model matches PyTorch to the fp32 floor.
- S4D seed 30485 runs: d64_seq256 79.60, d108_seq256 80.85 percent.
- Galaxy CNN 61K: not yet measured on the test set (pending).

Numbers are reproduced by scripts/run_all_benchmarks.sh. The full history and the
d64/d108 native and VLEN 512 study are in results/S4-Optimization.xlsx.
