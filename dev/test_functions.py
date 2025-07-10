import torch


def bumb(s, c=1):
    return torch.where(
        torch.abs(s) < c,
        1 / c * (1 / ((s / c)**2-1)).exp(),
        0)


def Gauss(s):
    return (-0.5 * s**2).exp()


def inverse_logarithmic(s, c=1):
    return 1 / (c**2 + (1 + s**2).log()).sqrt()


def inverse_multi_quadric(s, c=1):
    return 1 / (c**2 + s**2).sqrt()


def logarithmic(s):
    out = s.abs().log()
    if isinstance(out, torch.Tensor):
        out = torch.maximum(
            out, torch.tensor(-10.0, device=s.device, dtype=torch.float)
        )
    return out


def multi_quadric(s, c=1):
    return (c**2 + s**2).sqrt()


def Riesz(s, r=1):
    return -(s.abs() ** r)


def thin_plate(s):
    out = s ** 2 * (s.abs()).log()
    if isinstance(out, torch.Tensor):
        out = torch.nan_to_num(out, nan=0.0)
    return out
