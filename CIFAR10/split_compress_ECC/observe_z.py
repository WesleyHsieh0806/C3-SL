import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
dump_path = './log/Lambda_0_Batch64_ep100'
file_path = os.path.join(dump_path, "features.pkl")

with open(file_path, "rb") as f:
    stored_dict = pickle.load(f)
z, recover_z = stored_dict["z"], stored_dict["recover_z"]
print(z.shape)
print(recover_z.shape)
