# Approximation error over different regularizations tau

# Dependencies
import gc
import torch
from dev.test_functions import Laplace, thin_plate
from simple_torch_NFFT import Fastsum
from test.wrapper import build_kernel_params
from dev.constants import DEVICE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# prepare GPU
print(DEVICE)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Paramters
F = Laplace
name = "Laplace"
F_name = "Laplace"

# F = thin_plate
# name = "TPS"
# F_name = "thin_plate"

# Hyperparamters
T = 1
L = 2**10
K = 2**8
taus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
bs = 2**10

# Slicing paramters
d = 1000
P = 1000
slicing_mode = "orthogonal"

# Test data
N = 10000
M = N
reps = 10
torch.manual_seed(111)
torch.cuda.manual_seed_all(111)
X = torch.randn((N, d, reps))
Y = torch.randn((M, d, reps))
Xw = torch.rand((N, reps))
scale = torch.median(torch.norm(X, dim=1)).item()
print("scale", scale)

aprx_err = torch.zeros(3, reps, len(taus))
for rep in tqdm(range(reps), desc="Outer"):
    # load tensor onto DEVICE
    x = X[:, :, rep].to(DEVICE).contiguous()
    y = Y[:, :, rep].to(DEVICE).contiguous()
    w = Xw[:, rep].to(DEVICE).contiguous()

    # compute exact kernel sum, naivly
    fastsum = Fastsum(d, kernel=F_name, device=DEVICE)
    KS_exact = fastsum.naive(x, y, w, scale)

    for method in tqdm(range(3), desc="Inner", leave=False):

        for tau_idx in range(len(taus)):
            tau = taus[tau_idx]

            # compute paramters for kernel inversion
            kernel_params = build_kernel_params(
                F, d, method, T, L, K, tau, bs)
            fastsum = Fastsum(d,
                              kernel="other",
                              device=DEVICE,
                              kernel_params=kernel_params,
                              n_ft=2*K,
                              slicing_mode=slicing_mode,
                              batch_size_P=1000,
                              batch_size_nfft=64)
            # compute kernel sums via slicing
            KS_sliced = fastsum(x, y, w, scale, P)
            # store relative L2 error
            aprx_err[method, rep, tau_idx] = (
                (KS_sliced - KS_exact).norm()/KS_exact.norm())


mn = torch.mean(aprx_err, dim=1)
sd = torch.std(aprx_err, dim=1)

# Save data
DATA = torch.cat([torch.tensor(taus)[None, :], mn, sd], axis=0)
header = "tau mn0 mn1 mn2 sd0 sd1 sd2"
np.savetxt(f'taus_{name}{P}.dat', DATA.T, header=header, comments="")


# Visualization
ls = ["-", "--", ":", "-."]
colors = ["teal", "mediumvioletred", "navy", "orange"]
methods = ["S-L2-H1", "F-L2-H1", "F-H1-H1"]

plt.figure(figsize=(6, 3))
for method in range(3):
    plt.fill_between(taus, (mn[method, :]-sd[method, :]).cpu(),
                     (mn[method, :] + sd[method, :]).cpu(),
                     alpha=0.5, label=methods[method],
                     color=colors[method])
    plt.plot(taus, mn[method, :].cpu(), color=colors[method])
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"relative $\mathcal{E}_{d,P}[f_K,F]$")
plt.xlabel(r"$\tau$")
plt.ylim([1e-3, 1e-0])
plt.title(f"{slicing_mode} {name} P={P}")
plt.legend()
plt.tight_layout()
plt.show()
