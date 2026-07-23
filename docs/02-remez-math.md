# Opt #2 — Taylor series → Remez minimax math

**5.74 B (from 6.87 B), and GELU ~3× cheaper.**

## The problem

The from-scratch `exp/sin/cos/tanh` used **Taylor series with a data-dependent
early-exit loop** ("keep adding terms until the term is below epsilon"). That means a
variable-length loop per call, and Taylor converges slowly away from 0, so it needed
many terms.

## The fix

Replace each with a **fixed-degree Remez (minimax) polynomial** — the polynomial that
minimizes the *maximum* error over the reduction interval, so far fewer terms reach the
same accuracy, with **no loop and no branches** (fully predictable, pipeline-friendly).

`my_exp` is the template:

```
x = n·ln2 + r,   r ∈ [-ln2/2, ln2/2]   =>   e^x = 2^n · e^r
```
- Range-reduce with a **Cody-Waite split** `ln2 = ln2_hi + ln2_lo` so `n·ln2_hi` is
  exact in fp32 across the whole domain.
- Evaluate `e^r` with a 6-coefficient Horner minimax polynomial.
- Build `2^n` by writing `n` straight into an fp32's exponent bits (no `pow`).

`my_tanh(x) = 1 − 2/(e^{2x}+1)` reuses `my_exp`. `sin/cos` use the same
reduce-then-minimax-polynomial structure.

The exact coefficients live in `galaxy_s4d.c` (`my_exp`, `my_sin`, `my_cos`).

## Effect

GELU (which calls `tanh`→`exp` per element) got ~3× cheaper, and the discretization
`exp/sin/cos` calls became fixed-cost.
