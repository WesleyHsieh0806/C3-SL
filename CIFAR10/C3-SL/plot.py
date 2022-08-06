import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
'''
* Read the accuracy from EEC_CCL, ECC, recon and split
'''
# 2
ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC_v2", "Resnet50_log", "Middle-2", "Compression2_Batch64_ep150")
with open(os.path.join(ECC_path, "test_accuracy.csv"), 'r') as f:
    test_acc_ECC_2 = list(map(float, f.readline().split(',')))[:100]
# 4
ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC_v2", "Resnet50_log", "Middle-2", "Compression4_Batch64_ep150")
with open(os.path.join(ECC_path, "test_accuracy.csv"), 'r') as f:
    test_acc_ECC_4 = list(map(float, f.readline().split(',')))[:100]
# 8
ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC_v2", "Resnet50_log", "Middle-2", "Compression8_Batch64_ep150")
with open(os.path.join(ECC_path, "test_accuracy.csv"), 'r') as f:
    test_acc_ECC_8 = list(map(float, f.readline().split(',')))[:100]
# 16
ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC_v2", "Resnet50_log", "Middle-2", "Compression16_Batch64_ep150")
with open(os.path.join(ECC_path, "test_accuracy.csv"), 'r') as f:
    test_acc_ECC_16 = list(map(float, f.readline().split(',')))[:100]
# 32
ECC_path = os.path.join(os.path.dirname(__file__), "..",
                        "split_compress_ECC_v2", "Resnet50_log", "Middle-2", "Compression32_Batch64_ep150")
with open(os.path.join(ECC_path, "test_accuracy.csv"), 'r') as f:
    test_acc_ECC_32 = list(map(float, f.readline().split(',')))[:100]

# 2
Bottle_path = os.path.join(os.path.dirname(
    __file__), "..", "BottleNet++", "Resnet50_log", "Middle-2", "B64_Compress2_Ep150")
with open(os.path.join(Bottle_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_Bottle_2 = list(map(float, f.readline().split(',')))[:100]
# Bottle_path = os.path.join(os.path.dirname(
#     __file__), "..", "BottleNet++", "Resnet50_log", "Middle-2", "B64_Compress4_Ep150")
# with open(os.path.join(Bottle_path, "test_accuracy_batch64.csv"), 'r') as f:
#     test_acc_Bottle_4 = list(map(float, f.readline().split(',')))[:100]
Bottle_path = os.path.join(os.path.dirname(
    __file__), "..", "BottleNet++", "Resnet50_log", "Middle-2", "B64_Compress8_Ep150")
with open(os.path.join(Bottle_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_Bottle_8 = list(map(float, f.readline().split(',')))[:100]
Bottle_path = os.path.join(os.path.dirname(
    __file__), "..", "BottleNet++", "Resnet50_log", "Middle-2", "B64_Compress16_Ep150")
with open(os.path.join(Bottle_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_Bottle_16 = list(map(float, f.readline().split(',')))[:100]
Bottle_path = os.path.join(os.path.dirname(
    __file__), "..", "BottleNet++", "Resnet50_log", "Middle-2", "B64_Compress32_Ep150")
with open(os.path.join(Bottle_path, "test_accuracy_batch64.csv"), 'r') as f:
    test_acc_Bottle_32 = list(map(float, f.readline().split(',')))[:100]


cen_path = os.path.join(os.path.dirname(
    __file__), "..", "centralized", "Resnet50_log", "Ep100")
with open(os.path.join(cen_path, "test_accuracy.csv"), 'r') as f:
    test_acc_cen = list(map(float, f.readline().split(',')))[:]
'''
* Plot accuracy
'''
# colors = cm.rainbow(np.linspace(0, 1, 2))
# plt.plot(range(1, len(test_acc_ECC_2)+1), test_acc_ECC_2,
#          color=colors[0], linestyle='--', label='ECC')
# plt.plot(range(1, len(test_acc_Bottle)+1), test_acc_Bottle,
#          color='k', linestyle='-', label='BotteNet++')
# plt.plot(range(1, len(test_acc_cen)+1), test_acc_cen,
#          color='b', linestyle='-', label='Centralized')
# plt.legend()
# plt.show()

print("ECC2 acc:", np.average(test_acc_ECC_2[-10:]))
print("ECC4 acc:", np.average(test_acc_ECC_4[-10:]))
print("ECC8 acc:", np.average(test_acc_ECC_8[-10:]))
print("ECC16 acc:", np.average(test_acc_ECC_16[-10:]))
print("ECC32 acc:", np.average(test_acc_ECC_32[-10:]))
print("BottleNet++ 2 acc:", np.average(test_acc_Bottle_2[-10:]))
# print("BottleNet++ acc:", np.average(test_acc_Bottle_4[-10:]))
print("BottleNet++ 8 acc:", np.average(test_acc_Bottle_8[-10:]))
print("BottleNet++ 16 acc:", np.average(test_acc_Bottle_16[-10:]))
print("BottleNet++ 32 acc:", np.average(test_acc_Bottle_32[-10:]))
print("Centralized acc:", np.average(test_acc_cen[-10:]))


# Plot Compression ratio vs Accuracy
ratio = np.array([2, 4, 8, 16])
log_ratio = np.log(ratio) / math.log(2)
ECC_acc = np.array([0.87179, 0.87094, 0.87077, 0.86846])
BottleNet_acc = np.array([0.8737, 0.8737, 0.87373, 0.87028])
Cen_acc = 0.87102
plt.plot(log_ratio, ECC_acc, color='b',
         linestyle='-', marker='o', label="Proposed C3-SL")
plt.plot(log_ratio, BottleNet_acc, color='b', marker='o',
         linestyle='--', label='BottleNet++')
plt.axhline(y=Cen_acc, color='k', linewidth=5.0,
            linestyle='--', label='Vanilla SL')
plt.xticks(log_ratio, labels=ratio)


plt.ylim([0.855, 0.885])
plt.grid()
plt.ylabel("Accuracy", fontsize=14)
plt.xlabel("Compression Ratio "+r'$R$', fontsize=14)
plt.legend()
plt.savefig("./Accuracy.png", bbox_inches='tight')
plt.show()
plt.close()
# Plot Memory & FLOps vs Compression Ratio
ratio = np.array([2, 4, 8, 16, 32])
log_ratio = np.log(ratio) / math.log(2)
ECC_Memory = np.array([8192, 16384, 32768, 65536, 131072])
BottleNet_Memory = np.array([9441792, 8394752, 4198912, 2100992, 1052032])
ECC_FLOPs = np.array([33554432, 33554432, 33554432, 33554432, 33554432])
BottleNet_FLOPs = np.array([37748736+37754880, 20971520+20976640,
                            10485760+10490368, 5242880+5247232, 2621440+2625224])

fig, ax = plt.subplots()
ax2 = ax.twinx()  # for double y axis

ECC_line, = ax.plot(log_ratio, ECC_Memory, color='tab:orange',
                    linestyle='-', marker='o', label="Proposed C3-SL Memory")
ax.plot(log_ratio, BottleNet_Memory, color='tab:orange', marker='o',
        linestyle='--', label='BottleNet++ Memory')

ECC_line2, = ax2.plot(log_ratio, ECC_FLOPs, color='green',
                      linestyle='-', marker='o', label="Proposed C3-SL FLOPs")
ax2.plot(log_ratio, BottleNet_FLOPs, color='green', marker='o',
         linestyle='--', label='BottleNet++ FLOPs')
plt.xticks(log_ratio, labels=ratio)


# plt.ylim([0.865, 0.876])
# plt.grid()
ax.set_ylabel("Memory", color="tab:orange")
ax.tick_params(axis='y', colors='tab:orange')  # Change color of y ticks
ax2.tick_params(axis='y', colors='green')  # Change color of y ticks
ax2.set_ylabel("FLOPs", color="green")
ax2.legend(loc='center', bbox_to_anchor=[0.75, 0.75])
ax.legend()
plt.show()
