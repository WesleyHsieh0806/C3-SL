import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from utils import

dump_path = './log/Lambda_0_Batch64_ep100'
file_path = os.path.join(dump_path, "features.pkl")

with open(file_path, "rb") as f:
    stored_dict = pickle.load(f)
z, recover_z = stored_dict["z"], stored_dict["recover_z"]
print(z.shape)
print(recover_z.shape)

# check for cross-correlation


def CrossCorrelation(z):
    z = z - z.mean(0)
    z = z/torch.norm(z, dim=0)
    return z.T @ z


CC_z = CrossCorrelation(z)
CC_recover_z = CrossCorrelation(recover_z)
print("MSR between Cross Correlation matrix:{:.5f}".format(
    torch.mean((CC_z-CC_recover_z)**2)))
