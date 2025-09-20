import gc
import torch
from dev.slicing import slicing
from dev.test_functions import (Gauss, Laplace, Riesz, thin_plate, logarithmic,
                                inverse_multi_quadric, multi_quadric,
                                bumb, inverse_logarithmic)
from dev.constants import SQRT_2, DEVICE
from simple_torch_NFFT import Fastsum
from tqdm import tqdm

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
d = 1000
slicing_mode = "orthogonal"
taus = [1e-7, 1e-7, 1e-5]

# d = 100
# slicing_mode = "distance"
# taus = [1e-6, 1e-6, 1e-4]

# Hyperparamters
reps = 10
T = 1
bs = 2**6
bs_mmd = 50
N, M, P = 10000, 10000, 5000
L, K = 2**10, 2**8
Functions = [Gauss, Laplace, Riesz, thin_plate, logarithmic,
             inverse_multi_quadric, lambda t: bumb(t, c=2),
             inverse_logarithmic, multi_quadric]
Function_names = ["Gauss", "Laplace", "energy", "thin_plate", "logarithmic",
                  "imq", "bumb",
                  "inv-log", "mq"]

torch.manual_seed(42)
X = torch.randn((N, d, reps), device=DEVICE)
Y = torch.randn((M, d, reps), device=DEVICE)
Xw = torch.rand((N, reps), device=DEVICE)
scale = torch.median(torch.norm(X, dim=1)).item()
print("scale", scale)
aprx_err = torch.zeros(len(Functions), reps, 4)


for F_idx in tqdm(range(len(Functions)), desc="Outer"):
    F = Functions[F_idx]
    for rep in tqdm(range(reps), desc="inner", leave=False):
        kernel_sum = exact_mmd(
            F, scale, X[:, :, rep], Y[:, :, rep], Xw[:, rep], bs_mmd)
        for method in range(3):
            tau = taus[method]
            kernel_params = build_kernel_params(
                F, d, method, T, L, K, tau, bs)
            fastsum = Fastsum(d, kernel="other", device=DEVICE,
                              kernel_params=kernel_params,
                              n_ft=2*K,
                              batch_size_P=100,
                              batch_size_nfft=100,
                              slicing_mode=slicing_mode)
            kernel_sum0 = fastsum(X[:, :, rep], Y[:, :, rep],
                                  Xw[:, rep], scale, P)
            aprx_err[F_idx, rep, method] = torch.linalg.norm(
                kernel_sum0 - kernel_sum) / torch.linalg.norm(kernel_sum)
        if Function_names[F_idx] == "imq":
            kernel_params3 = build_kernel_params_imq(d, K)
            fastsum3 = Fastsum(d, kernel="other", device=DEVICE,
                               kernel_params=kernel_params3,
                               batch_size_P=100,
                               batch_size_nfft=100,
                               n_ft=2*K, slicing_mode=slicing_mode)

            kernel_sum3 = fastsum3(X[:, :, rep], Y[:, :, rep],
                                   Xw[:, rep], scale, P)
        elif F_idx < 5:
            fastsum3 = Fastsum(d, kernel=Function_names[F_idx], device=DEVICE,
                               n_ft=2*K, slicing_mode=slicing_mode,
                               batch_size_P=100,
                               batch_size_nfft=100)
            kernel_sum3 = fastsum3(X[:, :, rep], Y[:, :, rep],
                                   Xw[:, rep], scale, P)
        else:
            kernel_sum3 = kernel_sum
        aprx_err[F_idx, rep, 3] = (torch.linalg.norm(kernel_sum3 - kernel_sum)
                                   / torch.linalg.norm(kernel_sum))


# Compute means and std dev
mn = torch.mean(aprx_err, dim=1)
sd = torch.std(aprx_err, dim=1)


# Display errors in Latex format
print("Function & S-L2-H1          & F-L2-H1              & F-H1-H1              & Direct         \\\\")
for F_id in range(len(Functions)):
    print(f"{Function_names[F_id]} ", end="")
    for method in range(4):
        print(" & $\\num{" + f"{mn[F_id, method].item():.2e}" + "} $", end="")
    print("\\\\")
