import torch
import matplotlib.pyplot as plt
import numpy as np  # for plotting index array

# Example data
N = 10
M = 100

# Simulate data: shape (N, M, 2) for two methods
x = torch.randn(N, M, 2)

# Compute means and std dev along axis=1 (the M dimension)
means = torch.mean(x, dim=1)          # shape (N, 2)
std_devs = torch.std(x, dim=1)        # shape (N, 2)

# x-axis indices
n = np.arange(N)

# Plot both methods
plt.plot(n, means[:, 0].numpy(), label='Method 1 Mean')
plt.fill_between(n, (means[:, 0] - std_devs[:, 0]), (means[:, 0] + std_devs[:, 0]), alpha=0.3)

plt.plot(n, means[:, 1].numpy(), label='Method 2 Mean')
plt.fill_between(n, (means[:, 1] - std_devs[:, 1]), (means[:, 1] + std_devs[:, 1]), alpha=0.3)

plt.xlabel('n')
plt.ylabel('Mean value')
plt.legend()
plt.show()
