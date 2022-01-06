import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.pyplot import cm
from numpy import core
ECC_Dir = os.path.join(os.path.dirname(__file__), 'log')
recon_Dir = os.path.join(os.path.dirname(
    __file__), '..', 'split_compress_recon', 'log')
split_Dir = os.path.join(os.path.dirname(
    __file__), '..', 'split', 'log')
colors = cm.rainbow(np.linspace(0, 1, 3))

# Rec loss of the training feature
with open(os.path.join(ECC_Dir, "Lambda_0_Batch64", 'train_rec_loss_64_0.0.csv'), 'r') as f:
    train_rec_00 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(ECC_Dir, "Lambda_5e-4_Batch64", 'train_rec_loss_64_0.0005.csv'), 'r') as f:
    train_rec_00005 = list(map(float, f.readline().split(',')))[:]
# with open(os.path.join(recon_Dir, "Lambda_0_Batch64", 'train_rec_loss_64_0.0.csv'), 'r') as f:
#     rec_train_rec_00 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(recon_Dir, "Lambda_5e-4_Log", 'train_rec_loss_64.csv'), 'r') as f:
    rec_train_rec_00005 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(split_Dir, "Gradient_Loss", 'train_rec_loss_64.csv'), 'r') as f:
    split_train_rec = list(map(float, f.readline().split(',')))[:]


plt.plot(range(1, len(train_rec_00005)+1), train_rec_00005,
         color=colors[0], linestyle='-', label='ECC Lambda:0.0005')
plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
         color=colors[0], linestyle='--', label='ECC Lambda:0.0')
plt.plot(range(1, len(rec_train_rec_00005)+1), rec_train_rec_00005,
         color=colors[1], linestyle='-', label='Rec Lambda:0.0005')
# plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
#          color=colors[1], linestyle='--', label='Rec Lambda:0.0')
plt.plot(range(1, len(split_train_rec)+1), split_train_rec,
         color=colors[2], linestyle='-', label='Split')
plt.legend()
plt.xticks(range(1, len(train_rec_00005)+1, 5))
plt.savefig(os.path.join(ECC_Dir, "MSE_loss.png"))
plt.close()

plt.plot(range(1, len(train_rec_00005)+1), train_rec_00005,
         color=colors[0], linestyle='-', label='ECC Lambda:0.0005')
plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
         color=colors[0], linestyle='--', label='ECC Lambda:0.0')
plt.plot(range(1, len(rec_train_rec_00005)+1), rec_train_rec_00005,
         color=colors[1], linestyle='-', label='Rec Lambda:0.0005')
# plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
#          color=colors[1], linestyle='--', label='Rec Lambda:0.0')
plt.legend()
plt.xticks(range(1, len(train_rec_00005)+1, 5))
plt.savefig(os.path.join(ECC_Dir, "MSE_loss2.png"))
plt.close()

'''
* Gradient reconstruction loss
'''
# Rec loss of the training feature
with open(os.path.join(ECC_Dir, "Lambda_0_Batch64", 'train_grad_rec_loss_64_0.0.csv'), 'r') as f:
    train_grad_rec_00 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(ECC_Dir, "Lambda_5e-4_Batch64", 'train_grad_rec_loss_64_0.0005.csv'), 'r') as f:
    train_grad_rec_00005 = list(map(float, f.readline().split(',')))[:]
# with open(os.path.join(recon_Dir, "Lambda_0_Batch64", 'train_grad_rec_loss_64_0.0.csv'), 'r') as f:
#     rec_train_rec_00 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(recon_Dir, "Lambda_5e-4_Log", 'train_grad_rec_loss_64.csv'), 'r') as f:
    rec_train_grad_rec_00005 = list(map(float, f.readline().split(',')))[:]
with open(os.path.join(split_Dir, "Gradient_Loss", 'train_grad_rec_loss_64.csv'), 'r') as f:
    split_grad_train_rec = list(map(float, f.readline().split(',')))[:]


plt.plot(range(1, len(train_grad_rec_00005)+1), train_grad_rec_00005,
         color=colors[0], linestyle='-', label='ECC Lambda:0.0005')
plt.plot(range(1, len(train_grad_rec_00)+1), train_grad_rec_00,
         color=colors[0], linestyle='--', label='ECC Lambda:0.0')
plt.plot(range(1, len(rec_train_grad_rec_00005)+1), rec_train_grad_rec_00005,
         color=colors[1], linestyle='-', label='Rec Lambda:0.0005')
# plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
#          color=colors[1], linestyle='--', label='Rec Lambda:0.0')
plt.plot(range(1, len(split_grad_train_rec)+1), split_grad_train_rec,
         color=colors[2], linestyle='-', label='Split')
plt.legend()
plt.xticks(range(1, len(train_grad_rec_00005)+1, 5))
plt.savefig(os.path.join(ECC_Dir, "MSE_grad_loss.png"))
plt.close()

plt.plot(range(1, len(train_grad_rec_00005)+1), train_grad_rec_00005,
         color=colors[0], linestyle='-', label='ECC Lambda:0.0005')
plt.plot(range(1, len(train_grad_rec_00)+1), train_grad_rec_00,
         color=colors[0], linestyle='--', label='ECC Lambda:0.0')
plt.plot(range(1, len(rec_train_grad_rec_00005)+1), rec_train_grad_rec_00005,
         color=colors[1], linestyle='-', label='Rec Lambda:0.0005')
# plt.plot(range(1, len(train_rec_00)+1), train_rec_00,
#          color=colors[1], linestyle='--', label='Rec Lambda:0.0')
plt.legend()
plt.xticks(range(1, len(train_grad_rec_00005)+1, 5))
plt.savefig(os.path.join(ECC_Dir, "MSE_grad_loss2.png"))
plt.close()
