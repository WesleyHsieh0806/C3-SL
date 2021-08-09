import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm

Dir = os.path.dirname(__file__)
split_compress_path = os.path.join(Dir, "log")
split_path = os.path.join(Dir, '..', 'split', 'log', "Gradient_Loss")
centralized_path = os.path.join(Dir, '..', 'centralized', 'log')

''' 
* Plot test accuracy
'''
with open(os.path.join(split_path, 'test_accuracy_batch64.csv')) as f:
    split_testacc_64 = list(map(float, f.readline().split(',')))[:26]
with open(os.path.join(split_compress_path, 'test_accuracy_32.csv')) as f:
    split_compress_testacc_32 = list(map(float, f.readline().split(',')))[:26]
with open(os.path.join(split_compress_path, 'test_accuracy_64.csv')) as f:
    split_compress_testacc_64 = list(map(float, f.readline().split(',')))[:26]
with open(os.path.join(split_compress_path, 'test_accuracy_128.csv')) as f:
    split_compress_testacc_128 = list(map(float, f.readline().split(',')))[:26]
with open(os.path.join(split_compress_path, 'test_accuracy_256.csv')) as f:
    split_compress_testacc_256 = list(map(float, f.readline().split(',')))[:26]
# with open(os.path.join(split_path, 'test_accuracy.csv')) as f:
#     split_testacc = list(map(float, f.readline().split(',')))
with open(os.path.join(centralized_path, 'test_accuracy.csv')) as f:
    cen_testacc = list(map(float, f.readline().split(',')))

plt.xlim([0, len(split_compress_testacc_32)+1])
plt.ylim([0.05, 1])
plt.xticks(range(1, len(split_compress_testacc_32)+1))
color = cm.rainbow(np.linspace(0, 1, 6))

plt.plot(range(1, len(split_compress_testacc_32)+1), split_compress_testacc_32, color=color[0],
         marker='o', linestyle='-', label='Batch 32')
plt.plot(range(1, len(split_compress_testacc_64)+1), split_compress_testacc_64, color=color[1],
         marker='o', linestyle='-', label='Batch 64')
plt.plot(range(1, len(split_compress_testacc_128)+1), split_compress_testacc_128, color=color[2],
         marker='o', linestyle='-', label='Batch 128')
plt.plot(range(1, len(split_compress_testacc_256)+1), split_compress_testacc_256, color=color[3],
         marker='o', linestyle='-', label='Batch 256')
plt.plot(range(1, len(split_testacc_64)+1), split_testacc_64, color=color[4], marker='o',
         linestyle='--', label='Without Compress 64')
plt.plot(range(1, len(cen_testacc)+1), cen_testacc, color=color[5], marker='o',
         linestyle='--', label='Centralized')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(os.path.join(split_compress_path, 'testacc.png'))
plt.close()
'''Transmission costs vs Test acc'''
# plt.xlim([0, len(split_compress_testacc_32)+1])
x = np.log(np.arange(1, len(split_compress_testacc_32)+1)*32*32*10000)
compressed_x = np.log(
    np.arange(1, len(split_compress_testacc_32)+1)*32*1*10000)
plt.ylim([0.2, 1])
color = cm.rainbow(np.linspace(0, 1, 5))

plt.plot(compressed_x, split_compress_testacc_32, color='r',
         marker='o', linestyle='-', label='Batch 32')
# plt.plot(range(1, len(split_compress_testacc_64)+1), split_compress_testacc_64, color=color[1],
#          marker='o', linestyle='-', label='Batch 64')
# plt.plot(range(1, len(split_compress_testacc_128)+1), split_compress_testacc_128, color=color[2],
#          marker='o', linestyle='-', label='Batch 128')
# plt.plot(range(1, len(split_compress_testacc_256)+1), split_compress_testacc_256, color=color[3],
#          marker='o', linestyle='-', label='Batch 256')
# plt.plot(range(1, len(split_testacc)+1), split_testacc,
#          color='r', marker='o', linestyle='--', label='Split')
plt.plot(x, cen_testacc[:len(compressed_x)], color='b', marker='o',
         linestyle='--', label='Centralized')
plt.legend()
plt.xlabel('Transmission costs(log)')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(os.path.join(split_compress_path, 'testacc_costs.png'))
plt.close()
'''
* Plot the accuracy of different cut layer
'''
labels = ['conv{}'.format(i) for i in range(1, 6)]
split_compress_cut = []
with open(os.path.join(split_compress_path, 'test_accuracy_cut1.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy_cut2.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy_cut3.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy_cut4.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy_cut5.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
color = cm.rainbow(np.linspace(0, 1, 5))

fig, ax = plt.subplots()
layers = ax.bar(x, split_compress_cut, width, label='Epoch30', color=color[0])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Split at different layer')
ax.set_xticks(x)
ax.set_ylim([0.05, 1])
ax.set_xticklabels(labels)
ax.legend()

# The label on each bar(the value)
ax.bar_label(layers, padding=3)
plt.savefig(os.path.join(split_compress_path, 'split_layer.png'))
plt.close()
