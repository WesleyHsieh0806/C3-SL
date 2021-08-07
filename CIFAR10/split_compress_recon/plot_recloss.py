import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.pyplot import cm
from numpy import core
Dir = os.path.join(os.path.dirname(__file__), "Lambda Log")
No_Compress_Dir = os.path.join(os.path.dirname(
    __file__), "..", "split", "log", "Gradient_Loss")
colors = cm.rainbow(np.linspace(0, 1, 3))
# Rec loss of The testing feature
with open(os.path.join(Dir, 'test_rec_loss_64_0.0.csv'), 'r') as f:
    test_rec_00 = list(map(float, f.readline().split(',')))[:20]
with open(os.path.join(Dir, 'test_rec_loss_64_0.0005.csv'), 'r') as f:
    test_rec_00005 = list(map(float, f.readline().split(',')))[:20]
with open(os.path.join(No_Compress_Dir, 'test_rec_loss_64.csv'), 'r') as f:
    test_rec_nocp = list(map(float, f.readline().split(',')))[:20]

# Rec loss of the training feature
with open(os.path.join(Dir, 'train_rec_loss_64_0.0.csv'), 'r') as f:
    train_rec_00 = list(map(float, f.readline().split(',')))[:20]
with open(os.path.join(Dir, 'train_rec_loss_64_0.0005.csv'), 'r') as f:
    train_rec_00005 = list(map(float, f.readline().split(',')))[:20]
with open(os.path.join(No_Compress_Dir, 'train_rec_loss_64.csv'), 'r') as f:
    train_rec_nocp = list(map(float, f.readline().split(',')))[:20]

# Rec loss of the training gradient
with open(os.path.join(Dir, 'train_grad_rec_loss_64_0.0.csv'), 'r') as f:
    train_grad_rec_00 = list(map(float, f.readline().split(',')))[:70]
with open(os.path.join(Dir, 'train_grad_rec_loss_64_0.0005.csv'), 'r') as f:
    train_grad_rec_00005 = list(map(float, f.readline().split(',')))[:70]
with open(os.path.join(No_Compress_Dir, 'train_grad_rec_loss_64.csv'), 'r') as f:
    train_grad_rec_nocp = list(map(float, f.readline().split(',')))[:70]

plt.plot(range(1, len(train_rec_nocp)+1), train_rec_nocp,
         color=colors[1], linestyle='-', label='Train Feature w/o Compress')
plt.plot(range(1, len(train_rec_00005)+1), train_rec_00005,
         color=colors[0], linestyle='-', label='Train Feature Lambda:0.0005')
plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
         color=colors[0], linestyle='--', label='Train Feature Lambda:0.0')
# plt.plot(range(1, len(test_rec_00005)+1), test_rec_00005,
#          color=colors[2], linestyle='-', label='Test Feature Lambda:0.0005')
# plt.plot(range(1, len(test_rec_00)+1), test_rec_00,
#          color=colors[2], linestyle='--', label='Test Feature Lambda:0.0')
plt.legend()
plt.xticks(range(1, len(train_rec_00005)+1))
plt.savefig(os.path.join(Dir, "MSE_loss.png"))
plt.close()

plt.plot(range(1, len(train_grad_rec_nocp)+1), train_grad_rec_nocp,
         color='r', linestyle='-', label='Train Gradient w/o Compress')
plt.plot(range(1, len(train_grad_rec_00005)+1), train_grad_rec_00005,
         color='g', linestyle='-', label='Train Gradient Lambda:0.0005')
plt.plot(range(1, len(train_grad_rec_00)+1), train_grad_rec_00,
         color='g', linestyle='--', label='Train Gradient Lambda:0.0')
plt.legend()
plt.xticks(range(1, len(train_grad_rec_00005)+1, 5))
plt.savefig(os.path.join(Dir, "MSE_grad_loss.png"))
plt.close()
