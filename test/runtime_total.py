# Runtime comparison for different data scales N

# Dependencies
import gc
import os
os.environ["TORCH_LOGS"] = "recompiles"  # needs to be set first
import torch
import numpy as np
from dev.test_functions import multi_quadric
from dev.constants import DEVICE
from dev.slicing import slicing
from simple_torch_NFFT import Fastsum
from test.wrapper import build_kernel_params
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import torch._dynamo # because we use different shapes
torch._dynamo.config.cache_size_limit = 64   # or higher if needed
torch._dynamo.reset()

# prepare GPU
print(DEVICE)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# Paramters
d = 1000
F = multi_quadric
F_name = "multi_quadric"

# Hyperparamters
T = 1
L = 2**10
K = 2**8
tau = 1e-7
method = 0
bs = 2**10

# Slicing paramters
P = 1000
slicing_mode = "orthogonal"

# Test data
reps = 14
N_max = 50000
N_list = torch.linspace(0, N_max, 11, dtype=torch.int)[1:]
times = torch.zeros(2, reps, len(N_list))
error = torch.zeros(reps, len(N_list))
torch.manual_seed(444)
torch.cuda.manual_seed_all(444)
X = torch.randn((N_list[-1], d, reps))
Y = torch.randn((N_list[-1], d, reps))
Xw = torch.rand((N_list[-1], reps))

# create Fastsum object
kernel_params = build_kernel_params(F, d, method, T, L, K, tau, bs)
fastsum_sliced = Fastsum(
    d,
    kernel="other",
    device=DEVICE,
    kernel_params=kernel_params,
    n_ft=2*K,
    slicing_mode=slicing_mode,
    batch_size_P=1000,
    batch_size_nfft=64
)


for N_idx in tqdm(range(len(N_list)-1, -1, -1), desc="Outer"):
    N = N_list[N_idx]

    for rep in tqdm(range(reps), desc="inner", leave=False):
        x = X[:N, :, rep].to(DEVICE).contiguous()
        y = Y[:N, :, rep].to(DEVICE).contiguous()
        w = Xw[:N, rep].to(DEVICE).contiguous()
        scale = torch.median(torch.norm(x, dim=1)).item()

        # compute the times for fastsummation
        torch.cuda.synchronize()
        start_fastsum = time()
        kernel_sum_sliced = fastsum_sliced(x, y, w, scale, P)
        torch.cuda.synchronize()
        end_fastsum = time()
        times[0, rep, N_idx] = end_fastsum - start_fastsum

        fastsum_keops = Fastsum(d, kernel="Gauss")  # will be overwritten
        fastsum_keops.basis_F = lambda x, scale: F(x / scale)

        # compute tiem for PyKeOps
        torch.cuda.synchronize()
        start_keops = time()
        kernel_sum_keops = fastsum_keops.naive(x, y, w, scale)
        torch.cuda.synchronize()
        end_keops = time()
        times[1, rep, N_idx] = end_keops - start_keops
        error[rep, N_idx] = ((kernel_sum_sliced-kernel_sum_keops).norm()
                             / kernel_sum_keops.norm())


time_mn = torch.mean(times[:, 4:, :], dim=1)
time_sd = torch.std(times[:, 4:, :], dim=1)

# Save data
DATA = torch.cat([torch.tensor(N_list)[None, :], time_mn, time_sd], axis=0)
header = "N mn0 mn1 sd0 sd1"
np.savetxt(f'runtime_{N_max}.dat', DATA.T, header=header, comments="")

plt.plot(N_list, time_mn[0], label="fastsum+slicing", color='blue')
plt.fill_between(N_list, time_mn[0]-time_sd[0], time_mn[0]+time_sd[0], color='blue', alpha=0.3)
plt.plot(N_list, time_mn[1], label="PyKeOps", color='orange')
plt.fill_between(N_list, time_mn[1]-time_sd[1], time_mn[1]+time_sd[1], color='orange', alpha=0.3)

plt.legend()
plt.title(f"Runtime {F_name} P={P}, method={method}")
plt.xlabel(r"$N$")
plt.ylabel("time in seconds")
plt.show()

error_mn = torch.mean(error, dim=0)
error_std = torch.std(error, dim=0)
plt.plot(N_list, error_mn, color='blue')
plt.fill_between(N_list, error_mn-error_std, error_mn+error_std, color='blue', alpha=0.3)
plt.title("Error with std")
plt.show()
