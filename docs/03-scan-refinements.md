# Opt #3–#5 — Scan refinements

Three small, exact tightenings of the recurrent scan. **5.74 B → ~3.09 B.**

## #3 — Drop the unused `Im(C·x)` (5.07 B)
The output only needs `y = D·u + 2·Re(C·x)`. The scan was computing the full complex
product `C·x` including its imaginary part, which is then thrown away. Computing only
the real part `Re(C·x) = C_r·x_r − C_i·x_i` removes one multiply and one add per state
per timestep. Bit-exact.

## #4 — Hoist loop-invariant loads (5.09 B at O0, big win at O2)
The per-channel discretization constants (`A_bar`, `B_bar`, and the strided `C` loads)
depend only on `(channel, state)` — **not** on the timestep. They were being re-read
from memory inside the 4096-iteration timestep loop. Hoisting them to per-channel
arrays computed once, before the timestep loop, removes that repeated memory traffic.

## #5 — Hand-structured scan body (3.09 B)
The complex state update was expressed so the compiler emits fused multiply-adds and
keeps the hot values in registers, instead of a generic `complex_mul` call per state.
This removes the function-call overhead from the innermost loop.

Each of these is a *correctness-preserving* refactor — same math, fewer instructions.
