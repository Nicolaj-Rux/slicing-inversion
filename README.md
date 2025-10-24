![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
 

# Numerical Methods for Kernel Slicing
In this library, we provide methods for inverting the slicing operator $\mathcal{S}_d: L^2([0,1]) \mapsto L^2([0,1])$ defined as


$$ \mathcal{S}f(s):= \int_0^1 f(ts)\varrho_d(t)\,\mathrm{d}t, \quad \text{where}\quad  \varrho_d(t):= c_d(1-t^2)^{(d-3)/2} \quad \text{and}\quad  c_d:=\frac{2\Gamma(\frac{d}{2})}{\sqrt{\pi}\Gamma(\frac{d-1}{2})}.$$

Illustration of a forward pass through the slicing operator $\mathcal{S}_d$ with $d=50$.
The left panel displays the original function $f:[0,1]\to \mathbb{R}$, while the right panel shows its transformation $F = \mathcal{S}_d f:[0,1]\to \mathbb{R}$.
<p align="center">
  <img src="https://github.com/Nicolaj-Rux/slicing-inversion/blob/main/forwardpass.gif" width="900" /> 
</p>

Given $F$ and $d$, our implementation computes $f$ in terms of $K$ Fourier coefficients by solving a regularized minimization problem of the following type

$$\arg\min_{a\in \mathbb{R}^{K}} \Vert \mathcal{S}_d f_a-F\Vert\quad \text{where } f_a(t)=\sum_{k=0}^{K-1} a_k \cos(\pi k t).$$

## Dependencies

This project requires the following Python (version 3.9.21) packages:

```/dev``` requires:
- [PyTorch](https://pytorch.org/) version 2.5.1
- [NumPy](https://numpy.org/) version 2.0.2

```/test``` additionally requires:
- [PyKeOps](https://www.kernel-operations.io/) version 2.3
- [simple_torch_NFFT](https://github.com/johertrich/simple_torch_NFFT)
- [tqdm](https://tqdm.github.io/) version 4.66.5
- [matplotlib](https://matplotlib.org/) version 3.9.2

 ## Installation
 To install package run:
 ```python
pip install git+https://github.com/Nicolaj-Rux/slicing-inversion
```
Create test file ```test.py``` with content:
```python
from  slicing_inversion.slicing import slicing
S = slicing(F= lambda s: s**2, d=10)
print(S.evaluate())
```
When running ```python test.py``` should return
```python
(2.5949782411771594e-06, 6.033632278442383, 14.8980712890625)
```
## How to use it
Import the class ```slicing``` as ```from slicing_inversion.slicing import slicing```.
Prepare a torch function ```F``` from $[0,1]\to\mathbb{R}$ and a dimension $d\ge 3$.
For example ```def F(s): return torch.exp(-s**2)``` and ```d=100```.
Then ```S = slicing(F=F, d=d)``` creates an instance of that class. To compute $f$ first call ```S.get_matrix()```.
This step computes the inversion matrix which is independent of $F$.
It will automatically be cahed, so that it can be reused again if required.
Call ```S.get_range_coef()``` to expand $F$ by some basis (depends on the ```method```, see section Hyperparamters) and then recover the $f$ in terms of the cosine coefficients with ```S.get_domain_coef()```.
Afterwards the coeficients are stored under ```S.a```.
The complete flow would look like
```python
from  slicing_inversion.slicing import slicing
import torch

def F(s): return torch.exp(-s**2)
d = 100

S = slicing(F= lambda s: s**2, d=10)
S.get_matrix()
S.get_range_coef()
S.get_domain_coef()
print(S.a) # Fourier coefficients of f
```

### Hyperparamters
The class comes with several default hyperparamters 
```python
method: int = 0,
T: float = 1.,
L: int = 2**10,
K: int = 2**8,
tau: float = 1e-6,
bs: int = 2**10
```
The first hyperparameter, `method`, can be chosen as $0$, $1$, or $2$. Further details on the methods can be found in the corresponding paper [*Numerical Methods for Kernel Slicing*](https://arxiv.org/abs/2510.11478). A scaling of $T=1$ usually works best. The number of quadrature points is set by $L$, and the number of Fourier coefficients used to recover $f$ is $K$. When modifying $L$ and $K$, ensure that $L > K$ to maintain numerical integration stability. The regularization strength is $\tau$, for which we found $10^{-6}$ generally performs best. For `method = 2`, a larger regularization around $10^{-4}$ may be preferable. The batch size `bs` can be reduced when using large $L$ and $K$ on a small GPU.

The density $\varrho_d$ vanishes near $1$ and concentrates near $0$ as $d \to \infty$. Therefore, the forward operator $\mathcal{S}_d$ has little impact on values close to $1$. As a result, we can focus the approximation of $f$ on the interval $[0,T]$ with $T < 1$. However, in practice this has little effect on accuracy, so that $T=1$ usually works just as well.

<p align="center">
  <img src="https://github.com/Nicolaj-Rux/slicing-inversion/blob/main/density.gif" width="900" /> 
</p>



## Mathematical Background

Given a function $F:[0,\infty)\to \mathbb{R}$ and points $x_1,\ldots, x_N\in \mathbb{R}^d$, $y_1,\ldots, y_M\in \mathbb{R}^d$, as well as weights $w_1,\ldots, w_N\in \mathbb{R}$, define

$$
s_m := \sum_{n=1}^N w_n F(\Vert x_n - y_m\Vert) \quad \text{for all } m=1,\ldots,M.
$$

If $N=M$, the naive computation of $s$ requires $\mathcal{O}(N^2)$ operations, which is infeasible for large $N$.  
The paper [*Fast Kernel Summation in High Dimensions via Slicing and Fourier Transforms*](https://epubs.siam.org/doi/10.1137/24M1632085) introduces a method to reduce this quadratic time complexity to linear by slicing the function $F$ into a one-dimensional function $f:[0,\infty)\to \mathbb{R}$, i.e., finding $f$ such that

$$
F(\Vert x\Vert) = \int_{\mathbb{S}^{d-1}} f(|\langle x,\xi\rangle|)\mathrm{d}\xi \quad \text{for all } x \in \mathbb{R}^d. \qquad (\star)
$$

Choosing $P$ slicing directions $\xi_1,\ldots,\xi_P \in \mathbb{S}^{d-1}$, the integral can be approximated as

$$
F(\Vert x\Vert) = \int_{\mathbb{S}^{d-1}} f(|\langle x,\xi\rangle|)\mathrm{d}\xi 
= \mathbb{E}_{\xi\sim \mathcal{U}_{\mathbb{S}^{d-1}}}[f(|\langle x, \xi\rangle|)] 
\approx \frac{1}{P} \sum_{p=1}^P f(|\langle x, \xi_p\rangle|).
$$

More information about good choices of slicing directions can be found in [*FAST SUMMATION OF RADIAL KERNELS VIA QMC SLICING*](https://openreview.net/pdf?id=iNmVX9lx9l).  
The $d$-dimensional kernel summation can then be reduced to $P$ one-dimensional kernel summations:

$$
s_m = \sum_{n=1}^N w_n F(\Vert x_n - y_m\Vert) 
\approx \sum_{n=1}^N w_n \frac{1}{P} \sum_{p=1}^P f(|\langle x_n - y_m, \xi_p\rangle|) 
= \frac{1}{P} \sum_{p=1}^P \sum_{n=1}^N w_n f(|\langle x_n, \xi_p\rangle - \langle y_m, \xi_p\rangle|).
$$

At first, this seems more complicated. The advantage is that the one-dimensional kernel sums

$$
s_m^p := \sum_{n=1}^N w_n f(|\langle x_n, \xi_p\rangle - \langle y_m, \xi_p\rangle|)
$$

can be computed in linear time using Fourier methods. Expand $f$ as a Fourier series with $K$ coefficients:

$$
f(|t|) \approx \sum_{k=-K}^K c_k \mathrm{e}^{2\pi \mathrm{i} k t}.
$$

Then, for fixed $p$, we can compute $s_m^p$ simultaneously for all $m=1,\ldots,M$ as

$$
s_m^p = \sum_{n=1}^N w_n f(|\langle x_n,\xi_p\rangle - \langle y_m,\xi_p\rangle|) 
\approx \underbrace{\sum_{k=-K}^K c_k \mathrm{e}^{ - 2\pi \mathrm{i} k\langle y_m,\xi_p\rangle}}_{\tilde s_k^p} 
\underbrace{\sum_{n=1}^N \mathrm{e}^{2\pi \mathrm{i} k \langle x_n,\xi_p\rangle}}_{\hat w_k^p}.
$$

Now, $\hat w_k^p$ can be computed efficiently using NFFT in $\mathcal{O}(N + K\log K)$ time.  
The multiplication of $c_k$ and $\hat w_k^p$ is $\mathcal{O}(K)$, and computing $\tilde s_m^p$ for all $m=1,\ldots,M$ is $\mathcal{O}(M + K\log K)$.  
Since this must be done for each slicing direction, the total time complexity is $\mathcal{O}(P(N + M + K\log K))$.  
Because $P$ and $K$ are independent of the data sizes $N$ and $M$, this represents a huge improvement over $\mathcal{O}(NM)$ when $K,P \ll N,M$.

As a prerequisite to applying this slicing method, the slicing transformation must be inverted, i.e., finding $f$ satisfying ($\star$).  
Interestingly, ($\star$) is equivalent to $\mathcal{S}_d[f] = F$.  
For some kernels such as the Riesz, Gaussian, or Laplace kernel, the sliced kernel $f$ is known.  
In this project, we focus on numerical methods to compute $f$ in terms of $K$ Fourier coefficients, given an arbitrary kernel $F$ in dimension $d$.

## Citation

This library was written by [Nicolaj Rux](https://scholar.google.com/citations?user=G8T1CF8AAAAJ&hl=en) in the context of fast kernel summations via slicing.
If you find it usefull, please consider to cite

```
@misc{RHN2025,
      title={Numerical Methods for Kernel Slicing}, 
      author={Nicolaj Rux and Johannes Hertrich and Sebastian Neumayer},
      year={2025},
      eprint={2510.11478},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2510.11478}, 
}
```

 or

```
@inproceedings{HJQ2025,
  title={Fast Summation of Radial Kernels via {QMC} Slicing},
  author={Johannes Hertrich and Tim Jahn and Michael Quellmalz},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=iNmVX9lx9l}
}
```

or

```
@article{H2024,
  title={Fast Kernel Summation in High Dimensions via Slicing and {F}ourier transforms},
  author={Hertrich, Johannes},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={6},
  number={4},
  pages={1109--1137},
  year={2024}
}
```
