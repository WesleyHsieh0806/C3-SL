import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

dir_path = './log/Ep100'
file_path = os.path.join(dir_path, "features.pkl")

with open(file_path, "rb") as f:
    stored_dict = pickle.load(f)
z, recover_z = stored_dict["z"], stored_dict["recover_z"]
test_z, test_recover_z = stored_dict["test_z"], stored_dict["test_recover_z"]
print(z.shape)
print(recover_z.shape)
print(test_z.shape)
print(test_recover_z.shape)

'''
* Define the data to be observed
'''
column = ["MSE between CrossCorrelation",
          "MSE between inter-sample correlation", "MSE between two features", "MSE CC (test)", "MSE SC (test)", "MSE (test)"]
data = []
# check for cross-correlation


def CrossCorrelation(z):
    z = z - z.mean(0)
    z = z/np.linalg.norm(z, axis=0)
    return z.T @ z


def SampleCorrelation(z):
    z = z - z.mean(1, keepdims=True)
    z = z/np.linalg.norm(z, axis=1, keepdims=True)
    return z @ z.T


CC_z = CrossCorrelation(z)
CC_recover_z = CrossCorrelation(recover_z)
print("MSE between Cross Correlation matrix:{:.5f}".format(
    np.average((CC_z-CC_recover_z)**2)))
data.append(np.average((CC_z-CC_recover_z)**2))

# Inter-sample correlation
SC_z = SampleCorrelation(z)
SC_recover_z = SampleCorrelation(recover_z)
print("MSE between Sample Correlation matrix:{:.5f}".format(
    np.average((SC_z-SC_recover_z)**2)))
data.append(np.average((SC_z-SC_recover_z)**2))

# MSE between two features
print("MSE between features:{:.5f}".format(
    np.average((z-recover_z)**2)))
data.append(np.average((z-recover_z)**2))

'''
* Test features
'''
test_CC_z = CrossCorrelation(test_z)
test_CC_recover_z = CrossCorrelation(test_recover_z)
print("MSE between Cross Correlation matrix:{:.5f}".format(
    np.average((test_CC_z-test_CC_recover_z)**2)))
data.append(np.average((test_CC_z-test_CC_recover_z)**2))

# Inter-sample correlation
test_SC_z = SampleCorrelation(test_z)
test_SC_recover_z = SampleCorrelation(test_recover_z)
print("MSE between Sample Correlation matrix:{:.5f}".format(
    np.average((test_SC_z-test_SC_recover_z)**2)))
data.append(np.average((test_SC_z-test_SC_recover_z)**2))

# MSE between two features
print("MSE between features:{:.5f}".format(
    np.average((test_z-test_recover_z)**2)))
data.append(np.average((test_z-test_recover_z)**2))
# output the observation into csv
dump_path = os.path.join("./log/Ep100/z_observation")
if not os.path.isdir(dump_path):
    os.makedirs(dump_path)

df = pd.DataFrame([data], columns=column)
df.to_csv(os.path.join(dump_path, "Z_observation.csv"))
