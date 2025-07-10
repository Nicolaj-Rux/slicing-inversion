import torch

# Set default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default dtype
DTYPE = torch.float32

# Precomputed constants
SQRT_2 = torch.tensor(2.0, dtype=DTYPE).sqrt().item()
PI = torch.tensor(torch.pi, dtype=DTYPE).item()
SQRT_PI = torch.tensor(torch.pi, dtype=DTYPE).sqrt().item()
