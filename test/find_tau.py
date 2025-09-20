import gc
import torch
from dev.slicing import slicing
from dev.test_functions import (Gauss, Laplace, Riesz, thin_plate, logarithmic, bumb)
from dev.constants import SQRT_2, DEVICE
from simple_torch_NFFT import Fastsum
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# prepare GPU
print(DEVICE)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# exact computation of MMD
def exact_mmd(F, scale, X, Y, Xw, bs_mmd):
    S = torch.zeros(Y.shape[0], device=DEVICE)
    for a in range(0, Y.shape[0], bs_mmd):
        b = min([a + bs_mmd, Y.shape[0]])
        S[a:b] = F(torch.norm(X[None, :, :]-Y[a:b, None, :], dim=2)/scale)@Xw
    return S


# kernel inversion method within fastsum implementation
def build_kernel_params(F, d, method, T, L, K, tau, bs):
    def fourier_fun(x, scale):
        x_pos = torch.abs(x)
        S = slicing(F=lambda t: F(t/(2*scale*T)), d=d,
                    method=method, T=T,
                    L=L, K=K, tau=tau, bs=bs)
        S.get_matrix()
        S.get_range_coef()
        S.get_domain_coef()
        a = S.a / 2
        a[0] *= SQRT_2
        a = torch.cat([a, torch.tensor([0.0], device=DEVICE)])
        f_hat = a[x_pos]
        return f_hat
    return dict(fourier_fun=fourier_fun)


# Paramters
F = thin_plate
name = "TPS"
d = 1000

# F = Laplace
# name = "Laplace"
# d = 1000

# Hyperparamters
T = 1
L = 2**10
K = 2**8
taus = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
bs = 2**6

# Slicing paramters
N, M = 1000, 1000
P = 5000
slicing_mode = "iid"
bs_mmd = 50
reps = 10

torch.manual_seed(42)
X = torch.randn((N, d, reps), device=DEVICE)
Y = torch.randn((M, d, reps), device=DEVICE)
Xw = torch.rand((N, reps), device=DEVICE)

scale = torch.median(torch.norm(X, dim=1)).item()
print("scale", scale)
aprx_err = torch.zeros(3, reps, len(taus))


for method in tqdm(range(3), desc="Outer"):
    for rep in tqdm(range(reps), desc="inner", leave=False):
        # compute exact kernel sum naivly
        kernel_sum = exact_mmd(
            F, scale, X[:, :, rep], Y[:, :, rep], Xw[:, rep], bs_mmd)
        for tau_idx in range(len(taus)):
            tau = taus[tau_idx]
            # compute paramters for kernel inversion
            kernel_params = build_kernel_params(
                F, d, method, T, L, K, tau, bs)
            fastsum = Fastsum(d, kernel="other", device=DEVICE,
                              kernel_params=kernel_params,
                              n_ft=2*K,
                              slicing_mode=slicing_mode)
            # compute kernel sums fast
            kernel_sum0 = fastsum(X[:, :, rep], Y[:, :, rep],
                                  Xw[:, rep], scale, P)
            # store relative L2 error
            aprx_err[method, rep, tau_idx] = torch.linalg.norm(
                kernel_sum0 - kernel_sum) / torch.linalg.norm(kernel_sum)


# compute means and std dev along axis=1 (the M dimension)
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
    plt.plot(taus, mn[method, :].cpu())
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"relative $\mathcal{E}_{d,P}[f_K,F]$")
plt.xlabel(r"$\tau$")
plt.legend()
plt.tight_layout()
plt.show()
