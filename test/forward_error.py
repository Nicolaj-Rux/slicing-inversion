# compute forward error for Laplace and bumb function

# Dependencies
import torch
from dev.slicing import slicing
from dev.test_functions import Laplace, bump
from dev.constants import DEVICE
from dev.quadrature import Quad
import matplotlib.pyplot as plt
from dev.auxiliary_functions import eval_cos_series
import numpy as np

# Paramters:
F = lambda s: bump(s, 0.5)
name = "bump"

F = Laplace
name = "Laplace"

# Hyperparamters:
d = 1000
T = 1.0
L = 2**10
K = 2**8
bs = 2**6
taus = [1e-8, 1e-8, 1e-6]


# Reference solution
s = torch.linspace(0, 1, 1001, device=DEVICE)
Fs = F(s)

# Visualization
ls = ["-", "--", ":", "-."]
colors = ["teal", "mediumvioletred", "navy", "orange"]
methods = ["S-L2-H1", "F-L2-H1", "F-H1-H1"]
plt.figure(figsize=(6, 3))

DATA = torch.zeros(1001, 4)
DATA[:, 3] = s.cpu()
quad_eval = Quad(2*L, d)  # twice as many quadrature points for evaluation

for method in range(3):
    f = slicing(F, d, method, T, L, K, taus[method], bs)
    f.get_matrix()
    f.get_range_coef()
    f.get_domain_coef()
    f.f = lambda t: eval_cos_series(f.a, t, f.T)
    f.F_rec = lambda s: quad_eval.RLFI(f.f, s)
    F_recs = f.F_rec(s)
    DATA[:, method] = (Fs - F_recs).abs().cpu()
    plt.plot(
        s.cpu(), DATA[:, method],
        label=methods[method]+rf" $\tau={taus[method]:.0e}$",
        linestyle=ls[method],
        color=colors[method]
    )

plt.legend()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.gca().yaxis.offsetText.set_fontsize(10)
plt.xlabel(r"$s$")
plt.ylabel(r"$|F(s)-F_K(s)|$")
plt.tight_layout()
plt.show()

# Save data
header = "m0 m1 m2 s"
np.savetxt(f'forward_{name}.dat', DATA, header=header, comments="")
