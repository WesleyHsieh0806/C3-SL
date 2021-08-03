import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm

Dir = os.path.dirname(__file__)
split_compress_recon_path = os.path.join(Dir,  "log")
split_compress_path = os.path.join(Dir, '..', 'split_compress', "log")
split_path = os.path.join(Dir, '..', 'split', 'log')

''' 
* Plot test accuracy
'''
with open(os.path.join(split_path, 'test_accuracy_batch64.csv')) as f:
    split_testacc_64 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(split_compress_path, 'test_accuracy_64.csv')) as f:
    split_compress_testacc_64 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(split_compress_recon_path, 'test_accuracy_64.csv')) as f:
    split_compress_testacc_recon_64 = list(
        map(float, f.readline().split(',')))[:]
# with open(os.path.join(split_path, 'test_accuracy.csv')) as f:
#     split_testacc = list(map(float, f.readline().split(',')))
# with open(os.path.join(centralized_path, 'test_accuracy.csv')) as f:
#     cen_testacc = list(map(float, f.readline().split(',')))

plt.xlim([0, len(split_compress_testacc_recon_64)+1])
plt.ylim([0.05, 1])
plt.xticks(range(1, len(split_compress_testacc_recon_64)+1))
# color = cm.rainbow(np.linspace(0, 1, 6))

plt.plot(range(1, len(split_testacc_64)+1), split_testacc_64, color='r',
         linestyle='-', label='Vanilla SL')
plt.plot(range(1, len(split_compress_testacc_64)+1), split_compress_testacc_64, color='g',
         linestyle='-', label='Compressed HD-SL')
plt.plot(range(1, len(split_compress_testacc_recon_64)+1), split_compress_testacc_recon_64, color='b',
         linestyle='-', label='Compressed HD-SL + Reconstruction Loss')
plt.xticks(range(1, len(split_testacc_64)+1, 5))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(os.path.join(split_compress_recon_path, 'testacc.png'))
plt.close()

'''Transmission costs vs Test acc'''
# plt.xlim([0, len(split_compress_testacc_32)+1])
x = np.log(np.arange(1, len(split_testacc_64) +
                     1).astype("float")*32*64*10000*2304)
compressed_x = np.log(
    np.arange(1, len(split_compress_testacc_64)+1).astype("float")*32*1*10000*2304)
compressed_rec_x = np.log(
    np.arange(1, len(split_compress_testacc_recon_64)+1).astype("float")*32*1*10000*2304)

plt.ylim([0.2, 1])
color = cm.rainbow(np.linspace(0, 1, 5))
plt.plot(compressed_x, split_compress_testacc_64, color='g',
         marker='o', linestyle='-', label='Compressed HD-SL')
plt.plot(x, split_testacc_64, color='r', marker='o',
         linestyle='--', label='Vanilla SL')
plt.plot(compressed_rec_x, split_compress_testacc_recon_64, color='b', marker='o',
         linestyle='-', label='Compressed HD-SL + Reconstruction Loss')
plt.legend()
plt.xlabel('Transmission costs(log)')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(os.path.join(split_compress_recon_path, 'testacc_costs.png'))
plt.close()
