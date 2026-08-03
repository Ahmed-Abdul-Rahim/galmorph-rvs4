# Opt #1 — O(L²) convolution → O(L·N) recurrent scan

**91.22 B → 6.87 B instructions (~13×).** The single biggest algorithmic win.

## The problem

The original S4D layer materialized a length-`L` convolution kernel and convolved it
against the input — an **O(L²)** operation. With `L = 4096`, that is ~8.4 million
multiply-adds *per channel per layer*, dominated by a deeply nested triangular loop.

## The fix

An S4D layer is a linear state-space recurrence and can be run as an **O(L·N)** scan
instead of a convolution. For each timestep we update a small `N = 32`-element complex
state per channel:

```
x_t = A_bar · x_{t-1} + B_bar · u_t          (x_{-1} = 0)
y_t = D·u_t + 2·Re( C · x_t )
```

- `A_bar`, `B_bar` come from discretizing the layer's continuous parameters
  (`log_dt`, `log_A_real`, `A_imag`) — computed **once per channel**, then reused for
  all 4096 timesteps.
- The state is 32 *complex* numbers (`D_STATE = 64` = 32 conjugate pairs; the `2·Re`
  accounts for both members of each pair).

This replaces `O(L²)` work with `O(L·N)` — for `L=4096, N=32` that is ~256× less
arithmetic — and removes the enormous branch/loop overhead of the convolution.

## Why it's exact

The recurrence is algebraically identical to the convolution (it *is* the closed-form
of the same linear system), so predictions are unchanged within fp32 tolerance.
