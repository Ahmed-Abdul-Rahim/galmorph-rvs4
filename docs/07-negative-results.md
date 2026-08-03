# Negative results — what we tried that did NOT help

Kept honestly, because a thorough search is what proves opt #14 is a real floor. Every
one of these validated as *correct*; they just cost **more** instructions.

| # | Idea | Result | Why it failed |
|--:|---|--:|---|
| 6 | RVV scan at `LMUL=2` (two 16-wide strips) | 3.24 B | Two reductions + reloaded constants; worse than one m4 group (opt #8). |
| 7 | Hand-write the whole 32-state loop in inline asm | 3.14 B | Tied the compiler's hands on loop control; body-only asm was better. |
| 9 | Vectorize GELU **inside opt #8** | 2.99 B | Fought the scan for registers, perturbing S4D upward. (Worked later as opt #14, after opt #12 freed registers.) |
| 10 | Unordered reduction (`vfredusum`) | 3.24 B | QEMU models ordered and unordered reductions as similarly expensive; no gain. |
| 11 | Vectorize the scan across **channels** (kill the per-timestep reduction) | 3.51 B | The 32-state batch no longer fits in registers; the added state memory traffic outweighed the saved reductions. |
| 15 | Explicitly hoist `D[h]` out of the timestep loop | 1.83 B | Perturbed codegen slightly the wrong way; the compiler already handled it. |
| 16 | Replace `vfredosum` with a `vslidedown`+`vfadd` **tree reduction** | 2.62 B | `vslidedown` is *also* a cross-lane op QEMU models per-element (~900 each); 5 of them cost more than one `vfredosum`. |
| 17 | Reduce via **memory + scalar sum** | 2.02 B | GCC re-vectorized the scalar sum back into a `vfredosum`. |
| 18 | #17 + `no-tree-vectorize` to force the sum scalar | 2.01 B | Confirmed the cost was **not** the reduction — see methodology. |

The big lesson from #16–#18: the per-timestep cost is dominated by how QEMU models
**cross-lane vector operations**, and the ordered `vfredosum` is actually the *cheapest*
option available. Details in [`08-measurement-methodology.md`](08-measurement-methodology.md).
