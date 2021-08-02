import torch
from utils import circular_corr, circular_conv
W = torch.tensor([[1., 2], [3., 4]]).reshape([2,  2])
Key, V = circular_conv(W)
compress_V = torch.sum(V, dim=0, keepdims=True)
Recover_W = circular_corr(compress_V, Key)
print(W.shape)
print(Recover_W.shape)
