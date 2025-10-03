# Computing the approximation error for 7 RBFs

# Dependencies
import gc
import torch
from dev.test_functions import (Gauss, Laplace, thin_plate, logarithmic,
                                inverse_multi_quadric, multi_quadric, bump)
from dev.constants import DEVICE
from test.wrapper import build_kernel_params
from simple_torch_NFFT import Fastsum
from tqdm import tqdm

# prepare GPU
print(DEVICE)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# kernel inversion method within fastsum implementation for imq
def build_kernel_params_imq(d, K):
    def fourier_fun(x, scale):
        def f(t, scale): return (1+(t/scale)**2)**(-d/2)
        n_ft = x.shape[0]
        vect = f(torch.abs(x / n_ft), scale)
        vect_perm = torch.fft.ifftshift(vect)
        kernel_ft = 1 / n_ft * torch.fft.fftshift(torch.fft.fft(vect_perm))
        return kernel_ft
    return dict(fourier_fun=fourier_fun)


# Paramters
# d = 1000
# P = 1000

d = 100
P = 100

# Hyperparamters
T = 1
bs = 2**10
L = 2**10
K = 2**8
slicing_mode = "orthogonal"
taus = [1e-6, 1e-7, 1e-4]

# Test functions
Functions = [Gauss, Laplace, inverse_multi_quadric, thin_plate,
             logarithmic, multi_quadric, lambda t: bump(t, c=3)]
Function_names = ["Gauss", "Laplace", "imq", "thin_plate",
                  "logarithmic", "mq", "Bump"]

# Test data
N = 10000
M = N
reps = 10
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
X = torch.randn((N, d, reps))
Y = torch.randn((M, d, reps))
Xw = torch.rand((N, reps))
scale = torch.median(torch.norm(X, dim=1)).item()
print("scale", scale)


aprx_err = torch.zeros(len(Functions), reps, 4)
for F_idx in tqdm(range(len(Functions)), desc="Outer"):
    F = Functions[F_idx]
    torch.cuda.synchronize()
    for rep in tqdm(range(reps), desc="inner", leave=False):
        x = X[:, :, rep].to(DEVICE).contiguous()
        y = Y[:, :, rep].to(DEVICE).contiguous()
        w = Xw[:, rep].to(DEVICE).contiguous()

        # exact summation - will be modified in next lines
        fastsum = Fastsum(d, kernel="Gauss", device=DEVICE, slicing_mode="iid")
        fastsum.basis_F = lambda x, scale: Functions[F_idx](x / scale)
        KS_exact = fastsum.naive(x, y, w, scale)

        # methods F-L2-H1, S-L2-H1, S-H1-H1
        for method in range(3):
            tau = taus[method]
            kernel_params = build_kernel_params(F, d, method, T, L, K, tau, bs)
            fastsum = Fastsum(d, kernel="other", device=DEVICE,
                              kernel_params=kernel_params,
                              n_ft=2*K,
                              batch_size_P=1000,
                              batch_size_nfft=64,
                              slicing_mode=slicing_mode)
            KS_sliced = fastsum(x, y, w, scale, P)
            aprx_err[F_idx, rep, method] = (
                (KS_sliced - KS_exact).norm()/KS_exact.norm())

        # Direct method
        F_name = Function_names[F_idx]
        if F_name == "imq":
            kernel_params = build_kernel_params_imq(d, K)
            fastsum = Fastsum(d, kernel="other", device=DEVICE,
                              kernel_params=kernel_params,
                              batch_size_P=1000,
                              batch_size_nfft=64,
                              n_ft=2*K, slicing_mode=slicing_mode)

            KS_direct = fastsum(x, y, w, scale, P)
        elif F_name in ["Gauss", "Laplace", "thin_plate", "logarithmic"]:
            fastsum = Fastsum(d, kernel=Function_names[F_idx], device=DEVICE,
                              n_ft=2*K, slicing_mode=slicing_mode,
                              batch_size_P=1000,
                              batch_size_nfft=64
                              )
            KS_direct = fastsum(x, y, w, scale, P)
        else:
            KS_direct = KS_exact
        aprx_err[F_idx, rep, 3] = (KS_direct - KS_exact).norm()/KS_exact.norm()

print(f"d={d}, P={P}, taus={taus}, N={N}")

# Compute means and std dev
mn = torch.mean(aprx_err, dim=1)
sd = torch.std(aprx_err, dim=1)

# Display errors in Latex format
rel_sd = sd/mn
rel_sd = torch.nan_to_num(rel_sd, nan=0.0, posinf=0.0, neginf=0.0)
print("relative sd", rel_sd.max())

print("\n\n  MEAN\n")
# Display errors in Latex format
print("Function & S-L2-H1          & F-L2-H1              & F-H1-H1              & Direct         \\\\")
for F_id in range(len(Functions)):
    print(f"{Function_names[F_id]} ", end="")
    for method in range(4):
        print(" & $\\num{" + f"{mn[F_id, method].item():.2e}" + "} $", end="")
    print("\\\\")


import matplotlib.pyplot as plt
s = torch.linspace(-3, 3, 1001)
for f_id in range(len(Functions)):
    plt.plot(s, Functions[f_id](s), label=Function_names[f_id])
plt.ylim([-2, 3])
plt.show()
