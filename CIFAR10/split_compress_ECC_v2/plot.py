import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
'''
* Read the accuracy from EEC_CCL, ECC, recon and split
'''

ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC_v2", "Resnet50_log", "Middle-2", "Compression64_Batch64_ep150")
with open(os.path.join(ECC_path, "test_accuracy.csv"), 'r') as f:
    test_acc_ECC = list(map(float, f.readline().split(',')))[:100]


Bottle_path = os.path.join(os.path.dirname(
    __file__), "..", "BottleNet++", "Resnet50_log", "Middle-2", "B64_Compress64_Ep150")
with open(os.path.join(Bottle_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_Bottle = list(map(float, f.readline().split(',')))[:100]

cen_path = os.path.join(os.path.dirname(
    __file__), "..", "centralized", "Resnet50_log", "Ep100")
with open(os.path.join(cen_path, "test_accuracy.csv"), 'r') as f:
    test_acc_cen = list(map(float, f.readline().split(',')))[:]
'''
* Plot accuracy
'''
colors = cm.rainbow(np.linspace(0, 1, 2))
plt.plot(range(1, len(test_acc_ECC)+1), test_acc_ECC,
         color=colors[0], linestyle='--', label='ECC')
plt.plot(range(1, len(test_acc_Bottle)+1), test_acc_Bottle,
         color='k', linestyle='-', label='BotteNet++')
plt.plot(range(1, len(test_acc_cen)+1), test_acc_cen,
         color='b', linestyle='-', label='Centralized')
plt.legend()
plt.show()

print("ECC acc:", np.average(test_acc_ECC[-10:]))
print("BottleNet++ acc:", np.average(test_acc_Bottle[-10:]))
print("Centralized acc:", np.average(test_acc_cen[-10:]))
