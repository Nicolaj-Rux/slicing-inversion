import torch
from dev.slicing import slicing
from dev.test_functions import (Gauss, Laplace, Riesz, thin_plate, logarithmic, bumb)
from dev.constants import DEVICE
import matplotlib.pyplot as plt
from dev.auxiliary_functions import eval_cos_series
import numpy as np

print(DEVICE)

# Paramters:
F = lambda s: bumb(s, 0.5)
name = "bumb"
d = 1000
taus = [1e-9, 1e-9, 1e-8]

F = Laplace
name = "Laplace"
d = 1000
taus = [1e-7, 1e-7, 1e-5]

# F = thin_plate
# name = "TPS"
# d = 1000
# taus = [1e-7, 1e-7, 1e-5]


# Hyperparamters:
T = 1
L = 2**10
K = 2**8
bs = 2**4

# Reference solution
s = torch.linspace(0, 1, 1001, device=DEVICE)
Fs = F(s)

# Visualization
ls = ["-", "--", ":", "-."]
colors = ["teal", "mediumvioletred", "navy", "orange"]
methods = ["S-L2-H1", "F-L2-H1", "F-H1-H1"]
plt.figure(figsize=(6, 3))   # width=6 in, height=4 in

DATA = torch.zeros(1001, 4)
DATA[:, 3] = s.cpu()
for method in range(3):
    f = slicing(F, d, method, T, L, K, taus[method], bs)
    f.get_matrix()
    f.get_range_coef()
    f.get_domain_coef()
    f.f = lambda t: eval_cos_series(f.a, t, f.T)
    f.F_rec = lambda s: f.quad.RLFI(f.f, s)
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
plt.gca().yaxis.offsetText.set_fontsize(10)  # make the 10^-4 smaller if needed

# plt.yscale("log")
plt.xlabel(r"$s$")
plt.ylabel(r"$|F(s)-F_K(s)|$")
plt.tight_layout()   # adjust spacing so labels fit nicely
plt.show()

# Save data
header = "m0 m1 m2 s"
np.savetxt(f'forward_{name}.dat', DATA, header=header, comments="")
