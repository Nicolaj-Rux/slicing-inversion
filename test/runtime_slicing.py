# Runtime comparhison for slicing and fastsummation on either GPU or CPU

# Dependencies
import os
# decide for GPU or CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import gc
from dev.constants import DEVICE
# prepare GPU if aviable
print(DEVICE)
if DEVICE == "cuda":
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
from time import time
from dev.slicing import slicing
from dev.test_functions import multi_quadric
from simple_torch_NFFT import Fastsum
from test.wrapper import build_kernel_params
from tqdm import tqdm

# Hyperparamters
d = 1000
F = multi_quadric
F_name = "multi_quadric"
T = 1
L = 2**10
K = 2**8
tau = 1e-7
method = 0
bs = 2**10
reps = 14


# stopping runtime only for slicing
runtime = torch.zeros(reps, 2, 6)
for rep in tqdm(range(reps)):
    for method in range(1, -1, -1):
        # reset cash
        slicing.dict_spacial, slicing.dict_frequency = {}, {}
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time()
        S = slicing(F, d, method, T, L, K, tau, bs)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t1 = time()
        S.get_matrix()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t2 = time()
        S.get_range_coef()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t3 = time()
        S.get_domain_coef()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t4 = time()
        S.get_matrix()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t5 = time()
        runtime[rep, method, :] = torch.tensor([t1-t0 + t5-t4, t2-t1,
                                                t3-t2, t4-t3, 0, 0])

# Slicing paramters
P = 1000
slicing_mode = "orthogonal"

# Test data
N = 10000
times = torch.zeros(reps)
error = torch.zeros(reps)
torch.manual_seed(333)
if DEVICE.type == "cuda": torch.cuda.manual_seed_all(333)
X = torch.randn((N, d, reps))
Y = torch.randn((N, d, reps))
Xw = torch.rand((N, reps))

print(DEVICE)

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

# stopping runtime for fast summation
for rep in tqdm(range(reps)):
    x = X[:, :, rep].to(DEVICE).contiguous()
    y = Y[:, :, rep].to(DEVICE).contiguous()
    w = Xw[:, rep].to(DEVICE).contiguous()
    scale = torch.median(torch.norm(x, dim=1)).item()
    for method in range(1, -1, -1):

        # compute the times for fastsummation
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        start_fastsum = time()
        kernel_sum_sliced = fastsum_sliced(x, y, w, scale, P)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        end_fastsum = time()
        runtime[rep, method, 4] = end_fastsum - start_fastsum

        fastsum_keops = Fastsum(d, kernel="Gauss")  # will be overwritten
        fastsum_keops.basis_F = lambda x, scale: F(x / scale)

        # compute tiem for PyKeOps
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        start_keops = time()
        kernel_sum_keops = fastsum_keops.naive(x, y, w, scale)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        end_keops = time()
        runtime[rep, method, 5] = end_keops - start_keops

# Compute means and std dev
mn = torch.mean(runtime[4:, :, :], dim=0)
sd = torch.std(runtime[4:, :, :], dim=0)

# Display errors in Latex format
rel_sd = sd/mn
rel_sd = torch.nan_to_num(rel_sd, nan=0.0, posinf=0.0, neginf=0.0)
print("maximal relative sd", rel_sd.max())

# Display errors in Latex format
print("Alg. & Dev.         & 2. Mat              & 3. Rhs              & 4. Solve     & Fastsum          & naive  \\\\")
for method in range(2):
    print(f"{method} & {DEVICE} ", end="")
    for k in range(1, 6):
        print(" & $\\num{" + f"{mn[method, k].item():.2e}" + "} $", end="")
    print("\\\\")

