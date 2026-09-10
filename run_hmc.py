import os
import torch

# 1. CRITICAL: Restrict PyTorch to 1 CPU thread per process immediately
# This must be at the very top of the script to prevent OpenMP deadlocks.
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_value
from torchdiffeq import odeint

# 2. Forward TRPL Solver
def forward_trpl(log_A, log_B, log_C, n0s, t, norm=True, log=True):
    A = 10**log_A
    B = 10**log_B
    C = 10**log_C
    n0s_val = 10**n0s

    def ode_func(time, n):
        return -A * n - B * (n**2) - C * (n**3)

    n_t = odeint(
        ode_func,
        n0s_val,
        t,
        method="dopri5",
        rtol=1e-6,
        atol=1e-6,
        options={"max_num_steps": 10_000},
    )
    trpl_signal = n_t**2

    if log:
        trpl_signal = torch.log(trpl_signal)
    if norm:
        trpl_signal = trpl_signal / trpl_signal[0:1, :]

    return trpl_signal

# 3. Pyro Probabilistic Model
def model(t, n0s, observed_signals=None):
    log_A = pyro.sample("log_A", dist.Normal(-4.0, 0.5))
    log_B = pyro.sample("log_B", dist.Normal(-19.0, 0.5))
    log_C = pyro.sample("log_C", dist.Normal(-37.0, 0.5))
    sigma = pyro.sample("sigma", dist.HalfNormal(0.0005))

    mu = forward_trpl(log_A, log_B, log_C, n0s, t, norm=True, log=True)

    with pyro.plate("data_time", mu.shape[0], dim=-2):
        with pyro.plate("data_fluences", mu.shape[1], dim=-1):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=observed_signals)


if __name__ == "__main__":
    print("Synthesising ground truth data...")
    t_eval = torch.tensor([0] + list(np.logspace(-1, 4, 99)), dtype=torch.float64)
    n0s_eval = torch.tensor([15, 16, 17, 18], dtype=torch.float64)

    true_log_A = torch.tensor(-4.0, dtype=torch.float64)
    true_log_B = torch.tensor(-19.0, dtype=torch.float64)
    true_log_C = torch.tensor(-37.0, dtype=torch.float64)

    true_signals = forward_trpl(true_log_A, true_log_B, true_log_C, n0s_eval, t_eval, norm=True, log=True)
    
    noise_level = 0.0005
    observed_data = true_signals + torch.randn_like(true_signals) * noise_level

    # Save plot instead of showing it to avoid blocking execution
    plt.figure(figsize=(8, 5))
    for i, n0 in enumerate(n0s_eval):
        plt.plot(t_eval.numpy(), observed_data[:, i].numpy(), label=f"n0 = 10^{int(n0)} cm⁻³")
    plt.xlabel("Time (ns)")
    plt.ylabel("Normalised Log(TRPL Signal)")
    plt.xscale("log")
    plt.legend()
    plt.title("Synthetic Noisy TRPL Data")
    plt.savefig("synthetic_data.png")
    plt.show()
    print("Saved 'synthetic_data.png'.")

    # 4. Set up parallel MCMC
    print("\nInitialising NUTS Kernel and starting MCMC...")
    init_vals = {
        "log_A": torch.tensor(-4.25, dtype=torch.float64),
        "log_B": torch.tensor(-18.60, dtype=torch.float64),
        "log_C": torch.tensor(-36.80, dtype=torch.float64),
        "sigma": torch.tensor(0.0005, dtype=torch.float64),
    }

    nuts_kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))

    mcmc = MCMC(
        nuts_kernel,
        num_samples=400,
        warmup_steps=100,
        num_chains=6,
        mp_context="spawn",
    )

    # Run sampling
    mcmc.run(t_eval, n0s_eval, observed_data)
    
    print("\nSampling Complete. Summary:")
    mcmc.summary()