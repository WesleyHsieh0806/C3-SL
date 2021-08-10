import numpy as np
import torch
from torch._C import dtype
from torch.functional import norm
from utils import circular_conv, circular_corr, normalize_for_circular
from math import sqrt
from scipy.linalg import circulant

hi = torch.arange(900, dtype=torch.float32).reshape(
    [1, 900])*5.0
norm_hi, STD, MEAN = normalize_for_circular(hi)

mean = 0
dim = 1
for i in range(1, len(hi.shape)):
    dim *= hi.shape[i]
std = (1/dim) ** (1/2)
Key = (torch.randn(hi.shape)*std + mean)
Key /= torch.norm(Key, dim=1, keepdim=True)
Key = Key.reshape([1, 1, Key.shape[0], -1])


# print(hi)


compressedV = circular_conv(hi, Key)
recover_hi = circular_corr(compressedV, Key)

MSE_loss = ((recover_hi - hi)**(2)).mean()
# print(recover_hi)
print(MSE_loss)

# print(hi)


compressedV = circular_conv(norm_hi, Key)
recover_hi = circular_corr(compressedV, Key)
recover_hi *= (STD*sqrt(recover_hi.shape[-1]))
recover_hi += MEAN
MSE_loss = ((recover_hi - hi)**(2)).mean()
# print(recover_hi)
print(MSE_loss)
