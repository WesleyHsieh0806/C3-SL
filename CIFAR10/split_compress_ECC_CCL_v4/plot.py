import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
'''
* Read the accuracy from EEC_CCL, ECC, recon and split
'''
CCL_path = os.path.join(os.path.dirname(__file__),
                        "log", "L0.1_B5.0_Batch64_Ep100")

with open(os.path.join(CCL_path, "test_accuracy_64_0.1.csv"), 'r') as f:
    test_acc_CCL = list(map(float, f.readline().split(',')))[:70]

ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC", "log", "Lambda_0_Batch64")
with open(os.path.join(ECC_path, "test_accuracy_64_0.0.csv"), 'r') as f:
    test_acc_ECC = list(map(float, f.readline().split(',')))[:]

rec_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC", "log", "Lambda_1.0_Batch64")
with open(os.path.join(rec_path, "test_accuracy_64_1.0.csv"), 'r') as f:
    test_acc_rec = list(map(float, f.readline().split(',')))[:]

split_path = os.path.join(os.path.dirname(
    __file__), "..", "split", "log", "Gradient_Loss")
with open(os.path.join(split_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_split = list(map(float, f.readline().split(',')))[:]

'''
* Plot accuracy
'''
colors = cm.rainbow(np.linspace(0, 1, 2))
plt.plot(range(1, len(test_acc_CCL)+1), test_acc_CCL,
         color=colors[0], linestyle='-', label='CCL (Lambda 0.1)')
plt.plot(range(1, len(test_acc_ECC)+1), test_acc_ECC,
         color=colors[0], linestyle='--', label='ECC')
plt.plot(range(1, len(test_acc_rec)+1), test_acc_rec,
         color=colors[1], linestyle='-', label='rec')
plt.plot(range(1, len(test_acc_split)+1), test_acc_split,
         color='k', linestyle='-', label='Split')
plt.legend()
plt.show()
