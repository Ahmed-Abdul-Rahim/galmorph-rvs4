# Opt #12 — Fold B̄ into C̄ (the big one)

**2.68 B → 1.87 B (−30%).** The largest single step after the recurrent rewrite, and it
came from a change-of-variables, not more code.

## The trick

The scan does, per state per timestep:
```
x_r = A_r·x_r − A_i·x_i + B_r·u          (a fused multiply-add on B)
x_i = A_r·x_i + A_i·x_r + B_i·u          (another one)
y  += 2·(C_r·x_r − C_i·x_i)
```

`B̄` is a fixed per-state constant. Substitute **`x = B̄ · x'`**. Because `A` is
diagonal (scalar per state), `A·B̄ = B̄·A`, so:

```
x'_t = A_bar · x'_{t-1} + u_t            (u real -> adds to the real part only, NO multiply)
y_t  = D·u_t + Re( C_bar · x'_t ),   where  C_bar = 2 · C · B̄   (precomputed once)
```

`B̄` **disappears from the scan entirely**: the state update loses its two `B·u` fused
multiply-adds, and `u_t` (real) is simply added to the real part. The factor `2` is
folded into `C_bar` at discretization time too.

## Why it won so much

Two effects stack:
1. **Fewer ops:** one fewer FMA per state per timestep.
2. **Register relief (the bigger effect):** opt #8 used *all 32* vector registers.
   Removing the two `B̄` vectors freed two register groups, so the per-timestep
   temporaries stop spilling to the stack — the whole inner loop now fits.

`C_bar = 2·C·B̄` is a complex multiply computed **once per (channel, state)** in the
discretization block — negligible cost, paid `H·N = 2048` times, versus the
`H·L·N = 8.4M` times the scan runs.

This optimization was suggested by our TA and is exactly the kind of "precompute the
constant matrices once" insight that pays off on the hot path.
