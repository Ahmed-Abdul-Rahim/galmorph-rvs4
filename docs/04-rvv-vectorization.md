# Opt #8 — RVV vectorized scan (resident constants, single reduction)

**3.09 B → 2.68 B.** The scan goes from scalar to RISC-V Vector (RVV 1.0).

## The idea

The inner loop over the 32 states is data-parallel, so vectorize it. At `VLEN = 256`
and `SEW = 32`, an **`LMUL = 4` (m4)** vector group holds exactly **32 float32
elements** — the whole state fits in one vector register group.

Per channel, the scan becomes:

```
load A_bar_r, A_bar_i, B_bar_r, B_bar_i, C_r, C_i  (once, resident in vector regs)
state vxr, vxi = 0                                  (resident across all timesteps)
for t in 0..4095:
    complex update  vxr,vxi = A_bar * (vxr,vxi) + B_bar * u_t     (RVV FMAs)
    term            vt      = C_r*vxr - C_i*vxi
    y_t             = D*u_t + 2 * reduce(vt)                       (one vfredosum)
```

## Two things that made it fast (learned the hard way)

1. **Constants + state stay resident** in vector registers across the entire 4096-step
   timestep loop — they are loaded once, not per timestep.
2. **One `m4` group, one reduction per timestep.** An earlier `LMUL = 2` two-strip
   version (with two reductions) was *slower*; consolidating to a single 32-wide group
   with a single `vfredosum.vs` won.

The kernel is written with RVV intrinsics under `#ifdef __riscv`, with a scalar C
fallback so the file still builds and runs (and is testable) on a normal host.

> Requires `VLEN = 256`. Run the simulator with `vlen=256` (matching the VeeR config)
> — see [`08-measurement-methodology.md`](08-measurement-methodology.md).
