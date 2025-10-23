# main file for slicing: computing f in terms of cosine coefs
# Dependencies
import torch
from typing import Callable

# Local imports
from .quadrature import Quad
from .constants import DEVICE, DTYPE, PI, SQRT_2
from .auxiliary_functions import eval_cos_series


class slicing:
    """Handles different methods for slicing."""
    dict_spacial, dict_frequency = {}, {}

    def __init__(
        self, F: Callable[[torch.Tensor], torch.Tensor],
        d: int,
        method: int = 0,
        T: float = 1.,
        L: int = 2**10,
        K: int = 2**8,
        tau: float = 1e-6,
        bs: int = 2**10
    ):
        self.F = F              # function in the range of S_d
        self.d = torch.tensor([d], device=DEVICE)  # dimension
        self.method = method    # method
        self.T = round(T, 5)    # significant interval length (rounded)
        self.L = L              # number of quadrature nodes
        self.quad = Quad(L, d)  # quadrature rule
        self.K = K              # number of domain coefficients
        self.tau = tau          # regularization parameter
        self.bs = bs            # batch size

    def get_matrix(self) -> torch.Tensor:
        """Returns a method-specific matrix."""
        if self.method == 0:
            dict_method = slicing.dict_spacial
        else:
            dict_method = slicing.dict_frequency

        key = (self.d.item(), self.T, self.L, self.K)

        if key in dict_method:   # matrix is already cached in dictionary

            self.X = dict_method[key]

        else:                     # Matrix is not cached - needs to be build
            if self.method == 0:  # spacial
                X = self.quad.build_spacial(self.K, self.bs, self.T)
                slicing.dict_spacial[key] = X
                self.X = X

            else:                 # frequency
                X = self.quad.build_frequency(self.K, self.bs, self.T)
                slicing.dict_frequency[key] = X
                self.X = X

        return self.X

    def get_range_coef(self) -> torch.Tensor:
        """Returns tensor of shape (self.K,): range coefficients."""
        if self.method == 0:      # compute F(x_l)sqrt(v_l)
            F_points = self.F(self.quad.p_glq)
            self.b = F_points*(self.quad.v_glq.sqrt())

        else:                     # Fourier coefficients <F, cos>
            x = torch.linspace(0, 1, 4*self.L, device=DEVICE, dtype=DTYPE)
            Fx = self.F(x)
            ext = torch.cat([Fx, Fx.flip(0)], dim=0)  # length 2L
            coef = torch.fft.fft(ext)[:self.L].real / (4*self.L)  # cosine part
            coef[0] /= SQRT_2
            self.b = coef

        return self.b

    def get_domain_coef(self) -> torch.Tensor:
        """Returns a tensor of shape (self.K,): domain coefficients."""
        DT = 1+(PI*torch.arange(self.K, device=DEVICE)/self.T)**2
        DT = DT.sqrt().diag()
        if self.method == 0:
            self.b_reg = torch.vstack([
                self.b.unsqueeze(1),
                torch.zeros((self.K, 1), device=DEVICE)])

            self.X_reg = torch.vstack([self.X.T, self.tau*DT])

        elif self.method == 1:               # L^2 and H^1 regularized
            self.b_reg = torch.vstack([
                self.b.unsqueeze(1),
                torch.zeros((self.K, 1), device=DEVICE)])

            self.X_reg = torch.vstack([self.X, self.tau*DT])

        elif self.method == 2:                  # H^1 and H^1 regularized
            D = (1+(PI*torch.arange(self.L, device=DEVICE))**2).sqrt()
            self.b_reg = torch.vstack([
                (D*self.b).unsqueeze(1),
                torch.zeros((self.K, 1), device=DEVICE)])

            self.X_reg = torch.vstack([(D[:, None]*self.X), self.tau*DT])

        else:
            raise ValueError("Invalid combination of method and mode")

        self.a = torch.linalg.lstsq(
            self.X_reg, self.b_reg, driver="gels")[0].squeeze()

        return self.a

    def evaluate(self) -> tuple:
        """Returns tuple (err_L2, a_tikh, a_sobl) of float.
        err_L2: L2 error of forward error on [0,1]
        a_tikh: Euclidean norm of coeficients a
        a_sobl: Sobolev norm of coefficients a"""

        # recover coeficients
        self.get_matrix()
        self.get_range_coef()
        self.get_domain_coef()
        quad_eval = Quad(2*self.L, self.d)
        self.f = lambda t: eval_cos_series(self.a, t, self.T)
        self.F_rec = lambda s: quad_eval.RLFI(self.f, s)

        # compute residuals
        N = 1000
        s = torch.linspace(0, 1, N, device=DEVICE)
        F_recs = torch.zeros_like(s, device=DEVICE)
        for a in range(0, N, self.bs):
            b = min(a+self.bs, N)
            F_recs[a:b] = self.F_rec(s[a:b])
        err = torch.abs(self.F(s) - F_recs)

        err_L2 = torch.mean(err**2).sqrt().item()
        a_tikh = torch.linalg.norm(self.a).item()
        a_sobl = torch.linalg.norm(
            self.a*(1+(PI*torch.arange(self.K, device=DEVICE)/self.T)**2)
            .sqrt()).item()

        return err_L2, a_tikh, a_sobl
