# wraps the fastsum module around our slicing module
from dev.slicing import slicing
from dev.constants import SQRT_2, DEVICE
import torch


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


# exact computation of MMD (naive)
def exact_mmd(F, scale, X, Y, Xw, bs_mmd):
    S = torch.zeros(Y.shape[0], device=DEVICE)
    for a in range(0, Y.shape[0], bs_mmd):
        b = min([a + bs_mmd, Y.shape[0]])
        S[a:b] = F(torch.norm(X[None, :, :]-Y[a:b, None, :], dim=2)/scale)@Xw
    return S
