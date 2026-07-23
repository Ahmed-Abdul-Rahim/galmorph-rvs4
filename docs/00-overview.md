# Optimization Study — Overview

This branch is the **single-file, most-optimized C implementation** of the S4D
galaxy-morphology classifier, plus the full story of how it got there.

## The model

A 64×64 grayscale image is classified into one of four morphologies
(**Round Elliptical, In-between Elliptical, Cigar-shaped Elliptical, Edge-on Disk**):

```
image -> Hilbert scan -> linear up-projection -> S4D -> GELU -> S4D -> GELU
      -> take-last-timestep -> linear head -> softmax
```

S4D is a diagonal Structured-State-Space layer. All transcendental math
(`exp/sin/cos/tanh/sqrt`) is implemented from scratch — there is **no libm**
dependency in the forward pass.

## Headline result

| Build | Dynamic instructions (least of `-O0/-O2/-O3`) | vs. original |
|---|--:|--:|
| Original (O(L²) convolution + Taylor math) | 91,215,510,431 | 1× |
| **This file (opt #14)** | **1,821,992,255** | **50.1×** |

Accuracy is preserved end-to-end: predictions match the PyTorch reference on every
labeled sample (see the validation section in the top-level `README.md`).

## The optimization ladder

Each step changed **one thing**, was re-measured across `-O0/-O2/-O3`, and had to
keep predicting correctly. The value logged is the least of the three builds.

| # | Change | Instructions | Note |
|--:|---|--:|---|
| — | Original: conv + Taylor | 91.22 B | O(L²) causal convolution |
| 1 | conv → **recurrent scan** | 6.87 B | O(L·N); [`01-recurrent-scan.md`](01-recurrent-scan.md) |
| 2 | Taylor → **Remez** math | 5.74 B | [`02-remez-math.md`](02-remez-math.md) |
| 3 | drop unused `Im(C·x)` | 5.07 B | [`03-scan-refinements.md`](03-scan-refinements.md) |
| 4–5 | hoist C loads, hand-tune scan | 3.09 B | [`03-scan-refinements.md`](03-scan-refinements.md) |
| 8 | **RVV vectorized scan** (m4, resident constants) | 2.68 B | [`04-rvv-vectorization.md`](04-rvv-vectorization.md) |
| 12 | **fold B̄ into C̄** | 1.87 B | −30%, the big one; [`05-fold-b-into-c.md`](05-fold-b-into-c.md) |
| 13 | inline math (one TU) | 1.83 B | [`06-inline-math-and-gelu.md`](06-inline-math-and-gelu.md) |
| **14** | **+ vectorized GELU** | **1.82 B** | **best (this file)** |

Plenty of things were tried that **did not** help — those are documented honestly in
[`07-negative-results.md`](07-negative-results.md), because the failures are what
proved opt #14 is a real floor and not an early stop.

How every number was measured (and a surprising QEMU vector-cost finding) is in
[`08-measurement-methodology.md`](08-measurement-methodology.md).
