import torch
from dev.constants import DEVICE, DTYPE, PI, SQRT_PI, SQRT_2
from dev.test_functions import (
    bumb, Gauss, inverse_logarithmic, inverse_multi_quadric,
    Laplace, logarithmic, multi_quadric, Riesz, thin_plate
)
from dev.auxiliary_functions import get_c, varrho, varrho_inv, eval_cos_series
from dev.quadrature import GLQ
from dev.chebyshev import (
    build_Chebyshev, load_chebyshev_dict, save_chebyshev_dict
)
from dev.slicing import slicing
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 2.5

print("\nTests for constants.py:")
print(f"Running on {DEVICE} with precision {DTYPE}")
print(f"constants PI={PI}, SQRT_PI={SQRT_PI} and SQRT_2={SQRT_2}")

print("\nTests for test_functions.py:")
s = torch.linspace(-3, 3, 601)
plt.plot(s, bumb(s), label="bumb")
plt.plot(s, Gauss(s), label="Gauss")
plt.plot(s, inverse_logarithmic(s), label="ilog")
plt.plot(s, inverse_multi_quadric(s), label="imq")
plt.plot(s, Laplace(s), label="Laplace")
plt.plot(s, logarithmic(s), label="log")
plt.plot(s, multi_quadric(s), label="mq")
plt.plot(s, Riesz(s), label="Riesz")
plt.plot(s, thin_plate(s), label="thin_plate")
plt.ylim([-2, 3])
plt.legend()
# plt.show()
plt.close()

print("\nTests for auxiliary_functions.py:")
d = 10*(torch.arange(100)+3)
plt.scatter(d, get_c(d), marker=".")
# plt.show()
plt.close()

d = torch.tensor([10, 100, 1000])
s = torch.tensor([1e-1, 1e-8])
print([[varrho(d_, varrho_inv(d_, s_)).item() for d_ in d] for s_ in s])

print("\nTests for quadrature.py:")
N = 2**10
glq = GLQ(N)
plt.scatter(glq.points.cpu(), N*glq.weights.cpu(), marker=".")
# plt.show()
plt.close()

print(f"Integral of bumb : {glq.integrate(bumb, -1, 1).item()} = 0.444")
print(f"Integral of Riesz: {glq.integrate(Riesz, 0, 1).item()} = -0.5")
def monomial(x, n): return x[:, None]**n[None, :]


n = torch.tensor([3, 5, 7, 9], device=DEVICE)
print(f"Integral of x^n  : {-glq.integrate(monomial, 0, 1, n)}")
print("Integral of x^n  : ([-0.2500, -0.1667, -0.1250, -0.1000])")

F = thin_plate
T = 2
K = 32
coef = glq.compute_cos_coef(F, K, T)
s = torch.linspace(0, T, 100, device=DEVICE)
plt.plot(s.cpu(), F(s).cpu(), label="Gauss")
plt.plot(s.cpu(), eval_cos_series(coef, s, T).cpu(), "--", label="cos")
plt.legend()
# plt.show()
plt.close()

parsevall = glq.parsevall(F, coef, T)
print(f"L^2={parsevall[0].item()}, ell^2={parsevall[1].item()}")

d = torch.tensor([1000], device=DEVICE)
def f_imq(t): return (1**2+t**2)**(-d/2)


s = torch.linspace(0, 1, 101, device=DEVICE)
y1 = inverse_multi_quadric(s)
y2 = glq.RLFI(f_imq, d, s)
plt.plot(s.cpu(), y1.cpu(), "-", label="imq")
plt.plot(s.cpu(), y2.cpu(), "--", label="S_d[f_imq]")
plt.legend()
# plt.show()
plt.close()

glq.build_Fourier(d, 2**4, 2**5, T)
glq.build_Bessel_H(d, 2**5, T)


print("\nTests for chebyshev.py")
chebyshev_dict = load_chebyshev_dict()
print(chebyshev_dict.keys())
build_Chebyshev(1000, 8, 2, 1)
save_chebyshev_dict(chebyshev_dict)

print("\nTests for slicing.py:")

F = slicing(F=Gauss, d=10, method=2, mode=1,
            T=1, L=2**10, K=2**8, J=2**8, tau=1e-6)
err_mse, a_norm_2, a_sob_2, res_2 = F.evaluate()
print(err_mse, a_norm_2, res_2)

print("Terminated succesfully")
