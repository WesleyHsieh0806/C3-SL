import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
'''
* Read the accuracy from EEC_CCL, ECC, recon and split
'''
CCL_path = os.path.join(os.path.dirname(__file__),
                        "Resnet50_log", "L0.1_B5.0_Batch64_Ep100")

with open(os.path.join(CCL_path, "test_accuracy.csv"), 'r') as f:
    test_acc_CCL = list(map(float, f.readline().split(',')))[:]
CCL_path = os.path.join(os.path.dirname(__file__),
                        "Resnet50_log", "Scheduler", "L0.1_B5.0_Batch256_Ep200")

with open(os.path.join(CCL_path, "test_accuracy.csv"), 'r') as f:
    test_acc_CCL_Scheduler = list(map(float, f.readline().split(',')))[:]


ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC", "Resnet50_log", "Scheduler", "Lambda_0_Batch256_ep100")
with open(os.path.join(ECC_path, "test_accuracy_256_0.0.csv"), 'r') as f:
    test_acc_ECC = list(map(float, f.readline().split(',')))[:]

rec_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC", "Resnet50_log", "Lambda_5_Batch64_ep100")
with open(os.path.join(rec_path, "test_accuracy_64_5.0.csv"), 'r') as f:
    test_acc_rec = list(map(float, f.readline().split(',')))[:]

split_path = os.path.join(os.path.dirname(
    __file__), "..", "split", "log", "ResNet_Ep100")
with open(os.path.join(split_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_split = list(map(float, f.readline().split(',')))[:]

cen_path = os.path.join(os.path.dirname(
    __file__), "..", "centralized", "Resnet50_log", "Ep100")
with open(os.path.join(cen_path, "test_accuracy.csv"), 'r') as f:
    test_acc_cen = list(map(float, f.readline().split(',')))[:]
'''
* Plot accuracy
'''
colors = cm.rainbow(np.linspace(0, 1, 2))
plt.plot(range(1, len(test_acc_CCL_Scheduler)+1), test_acc_CCL_Scheduler,
         color=colors[0], linestyle='-', label='CCL (Lambda 0.1)')
plt.plot(range(1, len(test_acc_ECC)+1), test_acc_ECC,
         color=colors[0], linestyle='--', label='ECC')
plt.plot(range(1, len(test_acc_rec)+1), test_acc_rec,
         color=colors[1], linestyle='-', label='rec')
plt.plot(range(1, len(test_acc_split)+1), test_acc_split,
         color='k', linestyle='-', label='Split')
plt.plot(range(1, len(test_acc_cen)+1), test_acc_cen,
         color='b', linestyle='-', label='Centralized')
plt.legend()
plt.show()

print("CCL acc:", np.average(test_acc_CCL[-10:]))
print("CCL Scheduler acc:", np.average(test_acc_CCL_Scheduler[-10:]))
print("ECC acc:", np.average(test_acc_ECC[-10:]))
print("rec acc:", np.average(test_acc_rec[-10:]))
print("Split acc:", np.average(test_acc_split[-10:]))
print("Centralized acc:", np.average(test_acc_cen[-10:]))
