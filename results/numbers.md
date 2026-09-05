# Measured numbers (for the paper)

Basis: RiscvMinorCPU, in order, 100 MHz, VLEN 256, 22 nm McPAT, fp fraction 0.60,
16 KiB L1, one inference. Energy = runtime dynamic power times execution time.
Committed = gem5 committed (retired) instructions. All values are reproduced by a
clean checkout with scripts/run_all_benchmarks.sh and the pipeline energy runner,
using the repository's McPAT converter. C models validate against an independent
NumPy forward pass to about 4e-05.

## Controlled experiment: fixed sequence length (256 tokens)
| model | d_model | layers | build | committed | time (s) | energy (mJ) | accuracy |
|-------|---------|--------|-------|-----------|----------|-------------|----------|
| d64_seq256  | 64  | 2 | RVV    | 6,275,503  | 0.0864 | 0.876 | 79.60% |
| d64_seq256  | 64  | 2 | Scalar | 12,468,631 | 0.2013 | 1.748 |        |
| d108_seq256 | 108 | 3 | RVV    | 19,948,310 | 0.2880 | 2.857 | 80.85% |
| d108_seq256 | 108 | 3 | Scalar | 33,053,634 | 0.5320 | 4.774 |        |

At a fixed 256 token sequence the larger model costs 3.18 times the instructions
and 3.26 times the energy of the smaller, for 1.25 points of accuracy. Isolates
width and depth from sequence length.

## S4D d64 native (RGB, 4096 tokens)
| build | committed | time (s) | energy (mJ) |
|-------|-----------|----------|-------------|
| RVV    | 58,481,032  | 0.7156 | 8.136 |
| Scalar | 157,853,559 | 2.5496 | 22.185 |

## RVV versus scalar (same model, same core)
| model | fewer instr | faster | less energy |
|-------|-------------|--------|-------------|
| d64_seq256  | 1.99x | 2.33x | 2.00x |
| d108_seq256 | 1.66x | 1.85x | 1.67x |
| d64_native  | 2.70x | 3.56x | 2.73x |

## Accuracy (GalaxyMNIST test set)
- S4D d108 (linear_d108_seed2): 82.2 percent, C matches PyTorch to the fp32 floor.
- Seed 30485 runs: d64_seq256 79.60, d108_seq256 80.85 percent.
- Galaxy CNN 61K: not yet measured on the test set.

## CNN versus S4D (note)
The Galaxy CNN 61K energy (about 8.2 mJ RVV, roughly 2.9 times S4D d108) was
measured with a different McPAT converter and is pending re-measurement with the
repository converter once the CNN weight baking script is wired in. Parameter
counts: CNN 61,044; S4D d108 76,360.

Full history and the earlier VLEN 512 study are in results/S4-Optimization.xlsx.
