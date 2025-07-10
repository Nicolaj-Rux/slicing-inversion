# Dependencies
import torch
from numpy.polynomial.legendre import leggauss  # Gaussian quadrature
from typing import Callable, Any

# Local imports
from .constants import DEVICE, DTYPE, SQRT_2, PI
from .auxiliary_functions import varrho


class GLQ:
    """Handles Gaussian Legendre Quaderature points and integration"""

    GLQ_dict = {}  # chaching quadrature points

    def __init__(self, L: int = 2**10):
        self.L = L                          # amount of quaderature points
        if L not in GLQ.GLQ_dict:
            GLQ.GLQ_dict[L] = self.build_GLQ()  # cache quadrature
        self.points = GLQ.GLQ_dict[L][0]        # load points
        self.weights = GLQ.GLQ_dict[L][1]       # load weights

    def build_GLQ(self):
        """Returns Gaussian Legendre Quaderature points"""
        x_np, w_np = leggauss(self.L)
        points = torch.tensor(x_np, dtype=DTYPE, device=DEVICE)
        weights = torch.tensor(w_np, dtype=DTYPE, device=DEVICE)
        return (points, weights)

    def integrate(
        self,
        f: Callable[..., torch.Tensor],
        a: float,
        b: float,
        *args: Any
    ) -> torch.Tensor:
        """
        Return parameter integral of f(x, *args) from a to b.
        The shape of the result matches the shape of the broadcasted *args.
        """
        x = (b - a) / 2 * self.points + (b + a) / 2
        w = (b - a) / 2 * self.weights
        f_vals = f(x, *args)
        reshaped_w = w.view(-1, *[1] * (f_vals.ndim - 1))
        return torch.sum(f_vals * reshaped_w, dim=0)

    def compute_cos_coef(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        K: int,
        T: float = 1.0
    ) -> torch.Tensor:
        """Returns tensor of shape K:
        cosine coefs ak = 2/T int_0^T f(x)cos(pi*k*x/T)dx k>=1
                     a0 = 2/T int_0^T f(x)/sqrt(2) dx"""
        Ks = torch.arange(K, device=DEVICE)

        def func(x, k):
            return f(T * x[:, None]) * (PI * k[None, :] * x[:, None]).cos()

        coef = 2*self.integrate(func, 0, 1, Ks)
        coef[0] /= SQRT_2
        return coef

    def parsevall(
        self, f: Callable[[torch.Tensor], torch.Tensor],
        coef: torch.Tensor, T: float = 1.0
    ) -> torch.Tensor:
        """Returns floats L2, ell2: L^2 norm of f and l^2 norm of coef."""
        def func(x): return f(T * x)**2
        L2 = 2 * self.integrate(func, 0, 1)
        ell2 = torch.sum(coef**2)
        return torch.tensor([L2, ell2], device=coef.device)

    def RLFI(
        self, f: Callable[[torch.Tensor], torch.Tensor],
        d: torch.Tensor, s: torch.Tensor, T: float = 1.
    ) -> torch.Tensor:
        """Returns tensor like s: Riemann Liouville Fractional Integral
                                                of f on s in dim d"""
        def f_varrho_d(t, s):
            t, s = t.view(-1, *([1] * s.ndim)), s.view(1, *s.shape)
            return f(t * s) * varrho(d, t)
        return self.integrate(f_varrho_d, 0, T, s)

    def build_Fourier(
        self, d: torch.Tensor,
        K: int, J: int, T: float = 1.
    ) -> torch.Tensor:
        """Returns tensor Matrix of shape (J,K): display matrix of S_d
                                            from cosine to cosine"""
        def f(t, j, k):
            t, j, k = t[:, None, None], j[None, :, None], k[None, None, :]
            tkj = t * k / T
            return ((tkj + j).sinc() + (tkj - j).sinc()) * varrho(d, t)
        Js, Ks = torch.arange(J, device=DEVICE), torch.arange(K, device=DEVICE)
        X = self.integrate(f, 0, 1, Js, Ks)
        X[0, :] /= SQRT_2
        X[0, 0] = 1.
        return X

    def build_Bessel_H(self, d: torch.Tensor, K: int, T: float = 1.):
        """Returns evaluation of S_d[cos(pi k x/T)](points)"""
        ks = torch.arange(K, device=DEVICE)[:, None]
        XK = 0.5 * (self.points + 1)[None, :] * ks

        def f(t, xk):
            t, xk = t[:, None, None], xk[None, :, :]
            return (PI * t * xk / T).cos() * varrho(d, t)
        H = self.integrate(f, 0, 1, XK)
        H[0, :] /= SQRT_2
        return H
