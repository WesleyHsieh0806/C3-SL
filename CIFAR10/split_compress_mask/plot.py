import matplotlib.pyplot as plt
import numpy as np
import os

Dir = os.path.join(os.path.dirname(__file__), 'log')
compress_Dir = os.path.join(os.path.dirname(
    __file__), '..', 'split_compress', 'log')
ideal_Dir = os.path.join(os.path.dirname(
    __file__), '..', 'split', 'log')

# Read accuracy of mask
mask_acc = []
for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    with open(os.path.join(Dir, 'test_accuracy_64_gamma{}.csv'.format(gamma))) as f:
        mask_acc.append(np.max(list(map(float, f.readline().split(",")))))

# Read ideal accuracy
with open(os.path.join(ideal_Dir, 'test_accuracy_batch64.csv')) as f:
    ideal_acc = max((map(float, f.readline().split(","))))

# Read compressed accuracy
with open(os.path.join(compress_Dir, 'test_accuracy_64.csv')) as f:
    compress_acc = max(map(float, f.readline().split(",")))

print(len(mask_acc))
print(ideal_acc)
print(compress_acc)

# plot the accuracy
x = np.arange(1, len(mask_acc)+1)*0.1
plt.plot(x, mask_acc, marker='o',
         linestyle='-', color='b', label='Mask')
plt.hlines(y=ideal_acc, xmin=x[0], xmax=x[-1],
           color='r', linestyle='--', linewidth=4, label='Split'
           )
plt.hlines(y=compress_acc, xmin=x[0], xmax=x[-1],
           color='g', linestyle='-', linewidth=3, label='Split Compress'
           )
plt.grid()
plt.title("Gamma vs Acc")
plt.legend(loc=4)
plt.show()
plt.savefig(os.path.join(Dir, 'gamma.png'))
plt.close()
