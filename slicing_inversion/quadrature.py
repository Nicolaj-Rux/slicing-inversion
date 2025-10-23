# handles quadrature and building matrices S and H
# Dependencies
import torch
from numpy.polynomial.legendre import leggauss  # Gaussian quadrature
from typing import Callable

# Local imports
from .constants import DEVICE, DTYPE, SQRT_2, PI
from .auxiliary_functions import varrho


class Quad:
    """Handles Gaussian Legendre Quaderature"""

    GLQ_dict = {}  # chaching quadrature points

    def __init__(self, L: int = 2**10, d: int = 3):
        self.L = L                          # amount of quaderature points
        self.d = d                          # underling dimension
        if L not in Quad.GLQ_dict:
            Quad.GLQ_dict[L] = self.build_GLQ()       # cache quadrature
        self.p_glq = Quad.GLQ_dict[L][0]              # load points
        self.v_glq = Quad.GLQ_dict[L][1]              # load weights
        varrho_d = varrho(torch.tensor([d], device=DEVICE), self.p_glq)
        self.w_rho = varrho_d*self.v_glq              # density weights

    def build_GLQ(self):
        """Returns Gaussian Legendre Quaderature points on [0,1]"""
        x_np, w_np = leggauss(self.L)
        points = (1+torch.tensor(x_np, dtype=DTYPE, device=DEVICE))/2
        weights = torch.tensor(w_np, dtype=DTYPE, device=DEVICE)/2
        return (points, weights)

    def RLFI(self, f: Callable[[torch.Tensor], torch.Tensor], s: torch.Tensor
             ) -> torch.Tensor:
        """Returns tensor like s: Riemann Liouville Fractional Integral
                                                of f on s in dim d"""

        t_times_s = self.p_glq.view(-1, *([1] * s.ndim)) * s.view(1, *s.shape)
        return (f(t_times_s)*self.w_rho.view(-1, *([1] * s.ndim))).sum(dim=0)

    def build_frequency(self, K: int, bs: int = 2**10, T: float = 1.
                        ) -> torch.Tensor:
        """Returns tensor Matrix of shape (L, K): display matrix of S_d
                                            from cosine to cosine"""
        j = torch.arange(self.L, device=DEVICE)[None, :, None]
        S = torch.zeros(self.L, K, device=DEVICE)
        for a in range(0, K, bs):
            b = min(a + bs, K)
            k = torch.arange(a, b, device=DEVICE)[None, None, :]
            x = self.p_glq[:, None, None] * k / T
            S[:, a:b] = (((x+j).sinc()+(x-j).sinc())*self.w_rho[:, None, None]
                         ).sum(dim=0)
        S[0, :] /= SQRT_2
        S[0, 0] = 1.
        return S

    def build_spacial(self, K: int, bs: int = 2**10, T: float = 1.
                      ) -> torch.Tensor:
        """Returns evaluation of S_d[cos(pi k x/T)](points)"""

        # build $Y_{m,l} = h_m(x_l)$ via quadrature
        Y = torch.zeros(K, self.L, device=DEVICE)
        for a in range(0, K, bs):
            b = min(a + bs, K)
            k = torch.arange(a, b, device=DEVICE)[:, None, None]
            x = k*self.p_glq[None, :, None]*self.p_glq[None, None, :]
            Y[a:b, :] = ((PI*x/T).cos()*self.w_rho[None, :, None]).sum(dim=1)
        Y[0, :] /= SQRT_2

        # build $G_{m, l}= h_m{x_l}sqrt(v_l)$
        G = Y*((self.v_glq)[None, :].sqrt())
        return G
