import torch

print(torch.__version__)

print(torch.backends.mps.is_available())

print(torch.tensor([1., 1., 1.], device='mps:0'))