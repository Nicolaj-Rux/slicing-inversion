import os
import torch
import sympy
from tqdm import tqdm
from .constants import DEVICE


def compute_Chebyshev_entry(
    d_val: int, T_val: float, k_val: int, j_val: int
) -> sympy.Basic:
    """Returns entry k_val, j_val of the inverse of the display matrix of
                                    S_d^{-1} from cosine to Chebyshev"""
    l, m = sympy.symbols("l m", integer=True)
    d, T, k, j = sympy.symbols("d T k j")
    if k_val == 0:
        if j_val == 0:
            return sympy.sqrt(2)

        term1 = j * (-1)**j * sympy.sqrt(2*sympy.pi) / sympy.gamma(d/2)
        term2 = (-1)**l * (4*T)**l * sympy.gamma(j+l) * sympy.gamma((d+l)/2)
        term2 /= (l+1) * sympy.gamma(j-l+1) * sympy.gamma(2*l+1)
        term2 /= sympy.gamma((l+1)/2)
        total_sum = sympy.Sum(term2, (l, 0, j))
        result = term1 * total_sum
        resultN = result.subs({d: d_val, T: T_val, k: k_val, j: j_val}).doit()
        return resultN

    if j_val == 0:
        return sympy.S.Zero

    term1 = (-1)**(j + k) * 2 * j * sympy.sqrt(sympy.pi) / sympy.gamma(d/2)
    term2 = (4*T)**l * (-1)**l * sympy.gamma(l+1) * sympy.gamma(j+l)
    term2 *= sympy.gamma((d+l)/2)
    term2 /= sympy.gamma(j-l+1) * sympy.gamma(2*l+1) * sympy.gamma((l+1)/2)
    term3 = (1-(-1)**(k*l)) * sympy.cos(sympy.pi*(l-1)/2) / (sympy.pi*k)**(l+1)
    term4 = (-1)**m / (sympy.gamma(l - 2*m) * (sympy.pi * k)**(2 * (m+1)))
    inner_sum = sympy.Sum(term4, (m, 0, sympy.floor(l/2) - 1))
    total_sum = sympy.Sum(term2 * (term3 + inner_sum), (l, 0, j))
    result = term1 * total_sum
    resultN = result.subs({d: d_val, T: T_val, k: k_val, j: j_val}).doit()
    return resultN


def build_Chebyshev(
    d: int, K: int, J: int, T: float = 1.
) -> torch.Tensor:
    """Returns inverse of the display matrix of S_d^{-1}
                            from cosine to Chebyshev"""
    X = torch.zeros(K, J, device="cpu")
    for k in tqdm(range(K)):
        for j in range(J):
            X[k, j] = float(compute_Chebyshev_entry(d, T, k, j).evalf(25))
    return X.to(DEVICE)


def load_chebyshev_dict():
    """
    Loads the chebyshev_dict.pt file from the same directory as this script,
    or returns an empty dictionary if the file does not exist or fails to load.
    """
    file_path = os.path.join(os.path.dirname(__file__),
                             '..', 'chebyshev_dicts', 'chebyshev_d=1000.pt')
    file_path = os.path.abspath(file_path)  # resolves full path
    try:
        if os.path.exists(file_path):
            return torch.load(file_path, weights_only=True)
        else:
            return {}
    except Exception as e:
        print(f"Warning: Failed to load chebyshev_dict.pt: {e}")
        return {}


def save_chebyshev_dict(chebyshev_dict):
    """ Saves the given dictionary to chebyshev_dict.pt
        in the same directory as this script."""
    file_path = os.path.join(os.path.dirname(__file__),
                             '..', 'chebyshev_dicts', 'chebyshev_d.pt')
    file_path = os.path.abspath(file_path)  # resolves full path
    try:
        torch.save(chebyshev_dict, file_path)
    except Exception as e:
        print(f"Error: Failed to save chebyshev_dict.pt: {e}")
