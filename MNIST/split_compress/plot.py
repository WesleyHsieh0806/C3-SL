import matplotlib.pyplot as plt
import numpy as np
import os

Dir = os.path.dirname(__file__)
split_compress_path = os.path.join(Dir, "log")
split_path = os.path.join(Dir, '..', 'split', 'log')
centralized_path = os.path.join(Dir, '..', 'centralized', 'log')

''' 
* Plot test accuracy
'''
with open(os.path.join(split_compress_path, 'test_accuracy.csv')) as f:
    split_compress_testacc = list(map(float, f.readline().split(',')))
with open(os.path.join(split_path, 'test_accuracy.csv')) as f:
    split_testacc = list(map(float, f.readline().split(',')))
with open(os.path.join(centralized_path, 'test_accuracy.csv')) as f:
    cen_testacc = list(map(float, f.readline().split(',')))

plt.xlim([0, len(split_compress_testacc)+1])
plt.ylim([0.94, 1])
plt.xticks(range(1, len(split_compress_testacc)+1))
plt.plot(range(1, len(split_compress_testacc)+1), split_compress_testacc, color='r',
         marker='o', linestyle='-', label='Compressed HD-SL')
# plt.plot(range(1, len(split_testacc)+1), split_testacc,
#          color='r', marker='o', linestyle='--', label='SL')
plt.plot(range(1, len(cen_testacc)+1), cen_testacc, color='b', marker='o',
         linestyle='-', label='Centralized')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(os.path.join(split_compress_path, 'testacc.png'))
plt.close()

x = np.log(np.arange(1, len(split_compress_testacc)+1)*32*32*10000)
compressed_x = np.log(np.arange(1, len(split_compress_testacc)+1)*32*10000)
plt.ylim([0.94, 1])
# plt.xticks(x)
plt.plot(compressed_x, split_compress_testacc, color='r',
         marker='o', linestyle='-', label='Compressed HD-SL')
# plt.plot(range(1, len(split_testacc)+1), split_testacc,
#          color='r', marker='o', linestyle='--', label='SL')
plt.plot(x, cen_testacc, color='b', marker='o',
         linestyle='-', label='Centralized')
plt.legend()
plt.xlabel('Transmission costs(log)')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(os.path.join(split_compress_path, 'testacc_costs.png'))
plt.close()
