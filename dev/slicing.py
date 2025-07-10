# Dependencies
import torch
from typing import Callable

# Local imports
from .quadrature import GLQ
from .constants import DEVICE, DTYPE, PI, SQRT_2
from .utils import find_sup_key
from .chebyshev import load_chebyshev_dict, build_Chebyshev
from .auxiliary_functions import eval_cos_series


class slicing:
    """Handles different methods for slicing."""
    glq_super = GLQ(2**10)
    dict_Chebyshev = load_chebyshev_dict()
    dict_Bessel_H, dict_Bessel, dict_Fourier = {}, {}, {}
    dict_method = [dict_Bessel_H, dict_Bessel, dict_Fourier, dict_Chebyshev]

    def __init__(
        self, F: Callable[[torch.Tensor], torch.Tensor], d: int,
        method: int = 2, mode: int = 1, T: float = 1., L: int = 2**10,
        K: int = 2**8, J: int = 2**8, tau: float = 1e-6
    ):
        self.F = F             # function in the range of S_d
        self.d = torch.tensor([d], device=DEVICE)
        self.method = method   # method: 1=Bessel, 2=Fourier, 3=Chebyshev
        self.T = round(T, 5)   # significant interval length (rounded)
        self.L = L             # number of quadrature nodes
        self.glq = GLQ(L)      # quadrature rule
        self.K = K             # number of domain coefficients
        self.J = J             # number of range coefficients
        self.tau = tau         # regularization parameter
        self.mode = mode       # mode: (0=none, 1=L^2, 2=H^1 QR)
    #                                  (3=L^2, 4=H^1 Cholesky)
        self.key = (self.d.item(), self.T, self.L, self.K, self.J)
        if self.method == 3:
            self.key = (self.d.item(), self.T, 0, self.K, self.J)

    def get_matrix(self) -> torch.Tensor:
        """Returns a method-specific matrix."""
        key_sup = find_sup_key(slicing.dict_method[self.method], self.key)

        if key_sup is not None:   # Matrix is already cached in dictionary
            if self.method == 1 or self.method == 2:
                self.X = (
                    slicing.dict_method[self.method][key_sup]
                    [:self.J, :self.K]
                    .to(device=DEVICE, dtype=DTYPE)
                )

            elif self.method == 3:
                self.X = (
                    slicing.dict_method[self.method][key_sup]
                    [:self.K, :self.J]
                    .to(device=DEVICE, dtype=DTYPE)
                )

            if self.method == 1:
                max_key_sup = (self.d.item(), self.T, self.L,
                               max(key_sup[-2], key_sup[-1]))
                self.H = (
                    slicing.dict_Bessel_H[max_key_sup]
                    [:self.J, :]
                    .to(device=DEVICE, dtype=DTYPE)
                )

        else:                     # Matrix is not cached - needs to be build
            if self.method == 1:  # Bessel
                maxKJ = max(self.K, self.J)
                H = self.glq.build_Bessel_H(self.d, maxKJ, self.T)
                new_key = (self.d.item(), self.T, self.L, max(self.K, self.J))
                slicing.dict_Bessel_H[new_key] = H
                X = 1/2 * (H * self.glq.weights[None, :]) @ H.T
                new_key = (self.d.item(), self.T, self.L, maxKJ, maxKJ)
                slicing.dict_Bessel[new_key] = X
                self.X = X[:self.J, :self.K]
                self.H = H[:self.J, :]

            if self.method == 2:  # Fourier
                X = self.glq.build_Fourier(self.d, self.K, self.J, self.T)
                slicing.dict_Fourier[self.key] = X
                self.X = X[:self.J, :self.K]

            if self.method == 3:  # Chebyshev
                X = build_Chebyshev(self.d.item(), self.K, self.J, self.T)
                slicing.dict_Chebyshev[self.key] = X
                self.X = X

        return self.X

    def get_range_coef(self) -> torch.Tensor:
        """Returns tensor of shape (self.J,): range coefficients."""
        if self.method == 1:      # compute <F,h_{T,k}>
            F_points = self.F(1/2*(self.glq.points+1))
            self.b = 1/2 * self.H @ (self.glq.weights * F_points)

        elif self.method == 2:    # Fourier coefficients <F, cos>
            self.b = self.glq.compute_cos_coef(self.F, self.J, 1)

        elif self.method == 3:    # Chebyshef coefficients <phi, cos>
            def phi(t): return self.F(((PI*t).cos() + 1) / 2)
            self.b = self.glq.compute_cos_coef(phi, self.J, 1)
            self.b[0] /= SQRT_2

        return self.b

    def get_domain_coef(self) -> torch.Tensor:
        """Returns a tensor of shape (self.K,): domain coefficients."""
        if self.method in (1, 2) and self.mode in (0, 1, 2):  # QR decomp
            if self.mode == 0:                                # unregularized
                self.b_reg = self.b
                self.X_reg = self.X

            elif self.mode == 1:                              # L^2 regularized
                self.b_reg = torch.vstack([
                    self.b.unsqueeze(1),
                    torch.zeros((self.K, 1), device=DEVICE)])

                self.X_reg = torch.vstack([
                    self.X,
                    self.tau * torch.eye(self.K, device=DEVICE)])

            else:                                             # H^1 regularized
                self.b_reg = torch.vstack([
                    self.b.unsqueeze(1),
                    torch.zeros((self.K, 1), device=DEVICE)])

                self.X_reg = torch.vstack([
                    self.X,
                    self.tau*torch.diag(torch.arange(self.K, device=DEVICE))])

            self.a = torch.linalg.lstsq(
                self.X_reg, self.b_reg, driver="gels")[0].squeeze()

        elif self.method == 1 and self.mode in (3, 4):        # Cholesky decomp
            assert self.K == self.J, "self.K != self.J (not quadratic)"

            if self.mode == 3:                                # L^2 regularized
                self.X_reg = self.X+self.tau*torch.eye(self.K, device=DEVICE)
            else:                                             # H^1 regularized
                self.X_reg = self.X + self.tau * torch.diag(
                    torch.arange(self.K, device=DEVICE))

            L = torch.linalg.cholesky(self.X_reg)
            y = torch.linalg.solve_triangular(
                L, self.b.unsqueeze(-1), upper=False)
            self.a = torch.linalg.solve_triangular(
                L.T, y, upper=True).squeeze()

        elif self.method == 3 and self.mode == 0:             # Direct
            self.a = self.X @ self.b

        else:
            raise ValueError("Invalid combination of method and mode")

        return self.a

    def evaluate(self) -> tuple:
        s = torch.linspace(0, 1, 1001, device=DEVICE)
        self.get_matrix()
        self.get_range_coef()
        self.get_domain_coef()
        self.f = lambda t: eval_cos_series(self.a, t, self.T)
        self.F_rec = lambda s: slicing.glq_super.RLFI(self.f, self.d, s, 1)
        err = torch.abs(self.F(s) - self.F_rec(s))
        err_mse = torch.mean(err**2).item()
        a_norm_2 = torch.linalg.norm(self.a).item()
        a_sob_2 = torch.linalg.norm(
            self.a * torch.arange(self.K, device=DEVICE)).item()
        if self.method in (1, 2):
            res_2 = torch.linalg.norm(self.X@self.a - self.b).item()
        else:
            res_2 = None
        return err_mse, a_norm_2, a_sob_2, res_2
