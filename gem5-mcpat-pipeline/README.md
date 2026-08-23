# gem5 + McPAT pipeline for galmorph-rvs4

This package runs your `bench` binary through gem5. It then feeds the
gem5 output into McPAT. McPAT reports power, area, and timing for the
core. The package covers both your RVV build and a new scalar build, so
you can compare the two.

Run this on your own WSL Ubuntu machine, not in a Claude session. All
paths below assume you copy this folder next to your `galmorph-rvs4-main`
repository.

## What gem5 and McPAT do, and do not do

- gem5 runs your compiled binary and records what the hardware did:
  cycle counts, instruction counts, cache activity.
- McPAT does not run any code. It takes gem5's numbers, plus a
  description of the core, and estimates power, die area, and timing
  for that core at a given process node.
- Neither tool "ports" your C code. Your binary stays exactly what
  `make bench` already produces.

## Before you start: three choices this package already made

You asked for scripts, an in-order CPU model, and both an RVV and a
scalar run. Inside "in-order," this package uses gem5's **MinorCPU**
model, not the simpler TimingSimpleCPU. Two reasons:

1. MinorCPU is a real staged pipeline, closer in spirit to a VeeR-class
   in-order core than TimingSimpleCPU's one-stage model.
2. MinorCPU reports more of the per-structure activity counts that
   McPAT wants. TimingSimpleCPU and AtomicSimpleCPU report much less,
   because gem5 itself does not track detailed pipeline structures for
   those simple models.

The trade-off: gem5's own worked example for RVV uses the
out-of-order O3CPU, not MinorCPU. Nobody has published a test showing
MinorCPU runs RVV code correctly. Try MinorCPU first. If it faults or
hangs on the RVV binary, switch to TimingSimpleCPU
(`--cpu timing` in the commands below) to check whether the problem is
RVV-in-general or RVV-in-MinorCPU. See "If something breaks" below.

## What is measured, and what is a stand-in

McPAT needs numbers gem5's simple CPU models do not always report
(process node, some cache and branch-predictor stats). Where a real
gem5 number was not available, this package uses a labelled stand-in
value instead of leaving a blank. Every stand-in is marked in
`mcpat/template_inorder_riscv.xml` with a `STATIC ASSUMPTION` comment,
and every fallback estimate used at conversion time is printed to your
terminal as a warning. Read those warnings. The two that matter most:

- **Process node: 22 nm.** Nothing in your repository names a real
  target chip, so this is a placeholder. Change
  `core_tech_node` near the top of the template if you have a real
  number.
- **Int/float instruction split: 50/50.** gem5's simple CPU models do
  not report this breakdown directly, so `gem5_to_mcpat.py` uses a
  fixed ratio. Your workload is float-heavy (S4D forward pass), so the
  real ratio is probably not 50/50. Pass `--fp-fraction` to
  `gem5_to_mcpat.py` if you get a better number, or run
  `scripts/find_stats.sh` on your stats.txt to look for a real
  breakdown for your gem5 version.

## Step 0: isolate this repo's Python prerequisites

Before touching gem5/McPAT, set up a dedicated environment for this
*repository's* own Python packages (`torch`, `numpy`, `einops` from the
`requirements.txt` at the repo root -- used by the accuracy/weight-export
scripts, not by anything below). Keeping these in their own environment
means `pip install` never has to touch, or fight with, your system Python.

With venv:
```
python3 -m venv ~/.venvs/galmorph-rvs4
source ~/.venvs/galmorph-rvs4/bin/activate
pip install -r ../requirements.txt
```
or with conda:
```
conda create -n galmorph-rvs4 python=3.11 -y
conda activate galmorph-rvs4
pip install -r ../requirements.txt
```

**Deactivate it again before Steps 1-3 and before running anything below
that touches gem5 or McPAT** (`conda deactivate`, or just open a fresh
shell if you used venv). gem5's build links against the apt packages
`01_install_deps.sh` installs (boost, protobuf, hdf5); conda in
particular ships its own copies of those same libraries and has already
been found to conflict with gem5's build in this project. gem5 itself
needs no pip packages for what this pipeline does (see the note in
`02_build_gem5.sh` if you're curious why `--break-system-packages` isn't
used anywhere here).

## Step by step

1. `./01_install_deps.sh`
2. `./02_build_gem5.sh` -- slow, expect 20 minutes to over an hour. Pins
   gem5 to the exact tagged release (`v25.1.0.1`) this pipeline has been
   validated against, rather than the moving `stable` branch, so a fresh
   clone on a different machine or a different day builds the same gem5.
3. `./03_build_mcpat.sh` -- pins McPAT to `v1.3.0` for the same reason.
4. Apply the scalar-build patch to your own copy of the repository, OR
   just copy `patched-source/main.c` and `patched-source/Makefile`
   over your repository's copies (they are your original files plus
   the `bench-scalar` target -- diff is in
   `patched-source/scalar-build.patch` if you want to review it
   first).
5. In your repository:
   ```
   make bench          # your existing RVV build
   make bench-scalar    # new: same binary, scalar reference path
   ```
6. Edit the `settings` block at the top of `scripts/run_both.sh`
   (paths to your gem5 build and your repository), then:
   ```
   ./scripts/run_both.sh
   ```
   This writes `m5out-rvv/stats.txt` and `m5out-scalar/stats.txt`.
7. `./scripts/find_stats.sh m5out-rvv/stats.txt` -- check the stat
   names it finds against `STAT_MAP` in `mcpat/gem5_to_mcpat.py`. Fix
   any entry that does not match your gem5 version.
8. `./scripts/run_mcpat.sh` -- converts both runs and calls McPAT
   twice.
9. `python3 scripts/compare_report.py` -- prints a short RVV-vs-scalar
   table (Area, Power, **simulated execution time, energy/inference**)
   and writes `comparison.md`.

## Timing: simSeconds vs hostSeconds -- always use simSeconds

Every timing and energy number this pipeline produces comes from gem5's
`simSeconds` stat, which `compare_report.py` reads straight out of
`m5out-*/stats.txt`. `simSeconds` is derived purely from the simulated
pipeline's cycle count and clock frequency, inside gem5's own
event-driven scheduler -- your host machine's speed never enters that
computation, so it is identical for the same binary + gem5 build +
config on any machine.

gem5 also writes `hostSeconds`, which is the real wall-clock time gem5
itself took to run on your machine -- that one *does* scale with host
speed. Never substitute `hostSeconds`, and never wrap `run_both.sh` in
an external `time`, for anything you plan to compare across systems or
report as "execution time" or "energy" -- either will silently make
your numbers stop matching someone else's run. `energy/inference` in
`compare_report.py` is `Runtime Dynamic power x simSeconds`.

## If something breaks

- **gem5 will not decode an RVV instruction under MinorCPU:** rerun
  step 6 with `CPU=timing ./scripts/run_both.sh` to confirm the RVV
  binary runs correctly under TimingSimpleCPU. If it does, the problem
  is specific to MinorCPU and RVV together; report the faulting
  instruction and stay on `--cpu timing` for now, noting in your write-up
  that the McPAT numbers then rest on a thinner set of gem5 stats.
- **McPAT rejects the XML:** McPAT's error names the exact tag. Open
  the matching tag in a sample file under your McPAT checkout's
  `ProcessorDescriptionFiles/` folder and compare. This template was
  built by hand from that schema, not copied from a working RISC-V
  example, because none exists publicly yet.
- **`gem5_to_mcpat.py` exits with an ERROR (not a warning):** it means
  `system.cpu.numCycles` or `system.cpu.numInsts` was not in your
  stats.txt at all. Run `scripts/find_stats.sh` and check your gem5
  build actually ran (a zero-byte or truncated stats.txt usually means
  gem5 crashed before finishing).

## Files in this package

```
01_install_deps.sh, 02_build_gem5.sh, 03_build_mcpat.sh   one-time setup
patched-source/            your main.c + Makefile, plus the diff, with
                            the bench-scalar target added
configs/se_config.py        gem5 SE-mode config (binary, cpu model, VLEN/ELEN,
                            clock, cache sizes all as flags, not buried constants)
scripts/run_both.sh         runs the RVV and scalar binaries through gem5
scripts/find_stats.sh       greps a stats.txt for the stat groups McPAT needs
mcpat/template_inorder_riscv.xml   the McPAT input file (read the comments)
mcpat/gem5_to_mcpat.py      fills the template from a stats.txt
scripts/run_mcpat.sh        converts both runs and calls the mcpat binary
scripts/compare_report.py   prints and saves the RVV-vs-scalar comparison,
                            including simSeconds-based exec time and energy
```
