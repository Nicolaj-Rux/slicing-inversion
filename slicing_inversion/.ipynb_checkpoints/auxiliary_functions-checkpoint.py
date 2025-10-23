import torch
from .constants import SQRT_2, SQRT_PI, PI


def get_c(d: torch.Tensor) -> torch.Tensor:
    """Returns RLFI normalization constant c_d for dimension d."""
    return 2. / SQRT_PI * ((d / 2).lgamma() - ((d - 1) / 2).lgamma()).exp()


def varrho(d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Returns RLFI density varrho_d(t)."""
    return torch.where(t <= 1, get_c(d) * (1 - t**2) ** ((d - 3) / 2),
                       torch.tensor(0., device=t.device, dtype=t.dtype))


def varrho_inv(d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Returns inverse of density varrho_d(s)."""
    return torch.where(
        d >= 4,
        (1 - (2 / (d - 3) * (s.log() - get_c(d).log())).exp()).sqrt(),
        torch.ones_like(s, device=s.device))


def eval_cos_series(
    a: torch.Tensor,
    x: torch.Tensor,
    T: float = 1.0
) -> torch.Tensor:
    """Returns cosine series: a0/sqrt(2)+sum_{k=1}^K a_k*cos(pi*k*x/T)"""
    K = torch.arange(a.shape[0], device=a.device).view(-1, *[1] * x.ndim)
    a_reshaped = a.view(-1, *[1] * x.ndim)
    cosine_terms = (PI * K * x / T).cos()
    series = torch.sum(a_reshaped * cosine_terms, dim=0)
    correction = (1 / SQRT_2 - 1) * a[0]
    return series + correction


def clenshaw(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Returns Chebyshev series sum_{j=0}^J a_j T_j(x)."""
    b2 = torch.zeros_like(x)
    b1 = torch.zeros_like(x)
    for aj in reversed(a[1:]):
        b0 = 2 * x * b1 - b2 + aj
        b2 = b1
        b1 = b0
    return x * b1 - b2 + a[0]
