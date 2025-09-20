from time import time
from dev.slicing import slicing
from dev.test_functions import Riesz
from dev.constants import DEVICE
import torch
from tqdm import tqdm

reps = 100

# warmup
runtime = torch.zeros(reps, 3, 5)
for rep in tqdm(range(reps)):
    for method in range(3):
        # reset cash
        slicing.dict_spacial,  slicing.dict_frequency = {}, {}
        t0 = time()
        S = slicing(Riesz, 1000, method, 1, 2**10, 2**8, 1e-7, 2**8)
        t1 = time()
        S.get_matrix()
        t2 = time()
        S.get_range_coef()
        t3 = time()
        S.get_domain_coef()
        t4 = time()
        S.get_matrix()
        t5 = time()


# stopping runtime
runtime = torch.zeros(reps, 3, 5)
for rep in tqdm(range(reps)):
    for method in range(2, -1, -1):
        # reset cash
        slicing.dict_spacial, slicing.dict_frequency = {}, {}
        t0 = time()
        S = slicing(Riesz, 1000, method, 1, 2**10, 2**8, 1e-7, 2**8)
        t1 = time()
        S.get_matrix()
        t2 = time()
        S.get_range_coef()
        t3 = time()
        S.get_domain_coef()
        t4 = time()
        S.get_matrix()
        t5 = time()
        runtime[rep, method, :] = torch.tensor([t1-t0, t2-t1,
                                                t3-t2, t4-t3, t5-t4])

# Compute means and std dev
mn = torch.mean(runtime, dim=0)
sd = torch.std(runtime, dim=0)

# Display errors in Latex format
print("Alg. & DEVICE & 1. Init.          & 2. Build              & 3. Range              & 4. Domain             & 2. Caching \\\\")
for method in range(3):
    print(f"{method} & {DEVICE} ", end="")
    for k in range(5):
        print(" & $\\num{" + f"{mn[method, k].item():.2e}" + "} $", end="")
    print("\\\\")
