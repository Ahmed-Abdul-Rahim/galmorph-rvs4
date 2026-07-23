# Opt #13–#14 — Inline math + vectorized GELU

**1.87 B → 1.82 B.** Small on top of opt #12, but they make the file self-contained and
finish the vectorization story. This is the final, best build.

## #13 — Inline the math into one translation unit (1.83 B)

`gelu` calls `my_tanh` (which calls `my_exp`) once *per element* — 262,144 times per
layer. When the math lived in a separate `math.c`, GCC could **not** inline across
translation units (no LTO), so every element paid a full function-call/return.

Fix: put the math functions in the **same translation unit** as their callers. Now
`-O2` inlines `my_tanh`/`my_exp` into the GELU loop and the discretization, removing the
per-element call overhead. This is why `galaxy_s4d.c` is a **single file** — it is not
just tidiness, it is the optimization.

## #14 — Vectorized GELU (1.82 B, best)

GELU is elementwise over `L·D = 262,144` values — embarrassingly parallel. It is
vectorized at `m4` (32 elements per pass) with **vectorized `exp`/`tanh`** helpers
(`v_exp_m4`, `v_tanh_m4`) that evaluate the same Remez polynomials across a whole
vector. A scalar fallback is kept under `#else`.

An earlier attempt to vectorize GELU (opt #9) *regressed*, because back then it fought
the S4D scan for registers. Only after opt #12 freed the S4D register file did GELU
vectorization stand on its own without perturbing the scan.

> Numerical note: the vector `exp` rounds with round-to-nearest-even while the scalar
> path used round-half-away, so probabilities can differ in ~the 5th decimal. The
> predicted class is unaffected.
