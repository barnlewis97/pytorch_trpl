# TRPL Bayesian Inference — PyTorch/Pyro Prototype

## What this is

This repo contains a Bayesian inference pipeline for fitting time-resolved
photoluminescence (TRPL) decay data to a recombination model:

```
dn/dt = -A·n - B·n² - C·n³
```

where `A`, `B`, and `C` are trap-assisted, radiative, and Auger recombination
coefficients, inferred via Hamiltonian Monte Carlo (NUTS) over synthetic
noisy decay curves at several initial carrier densities.

`run_hmc.py` implements this using **PyTorch + Pyro + torchdiffeq**. It
works, but this was an exploratory build to see whether a pure-PyTorch stack
could handle it reasonably. It can't, not competitively — see below. The
production version of this model lives in a companion implementation using
**JAX + NumPyro + diffrax**, which is both simpler to run and dramatically
faster.

## Why the JAX/NumPyro/diffrax version is the one to use

The bottleneck in this problem isn't the sampler — it's that every NUTS
leapfrog step requires a full forward ODE solve plus a gradient through
it, and that happens hundreds of times per sample, across hundreds of
samples, across multiple chains. That structure — the same small,
cheap computation repeated an enormous number of times — is exactly
the case JAX's compilation model is built for, and exactly the case
PyTorch's eager-mode execution is not.

Concretely:

- **No JIT compilation.** JAX traces the whole model (ODE solve → log
  density → gradient) once and compiles it into a single fused XLA kernel.
  Every leapfrog step then runs that compiled kernel with no further
  Python involvement. PyTorch's eager mode re-dispatches through Python
  for every individual tensor op, on every leapfrog step, every sample,
  every chain — and with an ODE this cheap per-step, that dispatch
  overhead dominates the runtime rather than the actual arithmetic.

- **`torchdiffeq`'s adaptive solver is a Python-level loop.** Its
  step-size adaptation logic runs in Python, calling out to small tensor
  ops one at a time. `diffrax` (JAX's ODE library) compiles the equivalent
  adaptive loop into a single XLA construct with no per-step Python
  round-trip, which matters a lot here — this decay is stiff-ish near
  `t=0`, so the solver needs many small internal steps to hit tolerance.

- **No vectorized chains.** Running multiple chains in Pyro means spawning
  separate OS processes (`mp_context="spawn"`), each reimporting the
  stack and paying inter-process communication overhead. NumPyro can
  batch chains as a `vmap`'d dimension inside a single process — no
  process-spawn cost, no serialization, and it can use all available
  cores as one vectorized computation instead of N independent ones.

- **Ecosystem churn.** Pyro's MCMC output requires `arviz` for standard
  diagnostics/serialization, and `arviz`'s Pyro converter has been broken
  or missing across recent releases (an `arviz_plots`/matplotlib
  incompatibility on one version, then a full 1.0 rewrite that dropped
  `from_pyro` entirely, while `from_numpyro` was ported over). NumPyro's
  arviz integration is first-class and has stayed stable through this,
  which isn't a coincidence — arviz's own maintainers clearly prioritize
  the JAX-based PPL ecosystem.

None of this is a knock on PyTorch generally — it's the right tool for
large, dense neural network workloads (transformers, CNNs, anything
dominated by a handful of huge matmuls), where per-op dispatch overhead
is negligible next to the actual compute. This particular problem —
gradient-based MCMC over a stiff, repeatedly-solved ODE — is just not
that workload.

## Status of this repo

`run_hmc.py` is left here as a working reference/comparison point, not
as the recommended way to run this inference. If you want to actually
fit TRPL data with this model, use the JAX/NumPyro/diffrax
implementation instead.

## Requirements (for this PyTorch version, if you do run it)

```
torch
pyro-ppl
torchdiffeq
numpy
matplotlib
arviz<1.0   # from_pyro was dropped in arviz's 1.0 rewrite
```
