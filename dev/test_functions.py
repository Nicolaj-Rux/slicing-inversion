# test functions
import torch
from pykeops.torch import LazyTensor


def Gauss(s, c=1):
    return (-0.5 * (s/c)**2).exp()


def Laplace(s, c=1):
    return (-(s*c).abs()).exp()


def Riesz(s, r=1):
    return -(s.abs() ** r)


def thin_plate(s, c=1):
    out = (c*s) ** 2 * ((c*s).abs()).log()
    if isinstance(out, torch.Tensor):
        out = torch.nan_to_num(out, nan=0.0)
    return out


def inverse_multi_quadric(s, c=1):
    return 1 / (c**2 + s**2).sqrt()


def logarithmic(s, c=1):
    out = (s*c).abs().log()
    if isinstance(out, torch.Tensor):
        out = torch.maximum(
            out, torch.tensor(-10.0, device=s.device, dtype=torch.float)
        )
    return out


def bump(s, c=1.0):
    r = 1-(s / c) ** 2
    if isinstance(s, LazyTensor):
        return r.ifelse((-1.0 / r).exp(), 0.0)
    else:
        return torch.where(r >= 0, (-1.0 / r).exp(), 0.0)


def multi_quadric(s, c=1):
    return -(c**2 + s**2).sqrt()


def inverse_logarithmic(s, c=1):
    return 1 / (c + (1 + s**2).log()).sqrt()
