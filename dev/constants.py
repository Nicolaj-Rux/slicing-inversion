# constants
import torch

# Set default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"  # uncomment to use cpu instead of GPU
# Set default dtype
DTYPE = torch.float32

# Precomputed constants
SQRT_2 = torch.tensor(2.0, dtype=DTYPE).sqrt().item()
PI = torch.pi
SQRT_PI = torch.tensor(torch.pi, dtype=DTYPE).sqrt().item()
