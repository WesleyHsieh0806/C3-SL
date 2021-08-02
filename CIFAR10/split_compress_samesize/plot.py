import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm

Dir = os.path.dirname(__file__)
split_compress_path = os.path.join(Dir, "log")
# split_path = os.path.join(Dir, '..', 'split', 'log')

'''
* Plot the accuracy of different cut layer
'''
labels = ['conv{}'.format(i) for i in range(1, 4)] + ['Centralized']
split_compress_cut = []
with open(os.path.join(split_compress_path, 'test_accuracy_cut1.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy_cut2.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy_cut3.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])
with open(os.path.join(split_compress_path, 'test_accuracy.csv')) as f:
    split_compress_cut.append(list(map(float, f.readline().split(',')))[-1])

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
color = cm.rainbow(np.linspace(0, 1, 4))

fig, ax = plt.subplots()
layers = ax.bar(x, split_compress_cut, width, label='Epoch30', color=color[0])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Split at different layer')
ax.set_xticks(x)
ax.set_ylim([0.05, 0.75])
ax.set_xticklabels(labels)
ax.legend()

# The label on each bar(the value)
ax.bar_label(layers, padding=3)
plt.savefig(os.path.join(split_compress_path, 'split_layer.png'))
plt.close()
