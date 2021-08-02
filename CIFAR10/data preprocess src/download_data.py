import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_datasets as tfds
"""
Download CIFAR Training data and Testing data
"""
# Construct a tf.data.Dataset
train, test = tfds.as_numpy(tfds.load('cifar10', split=['train', 'test'],
                                      data_dir='../CIFAR/', batch_size=-1))
print("Size of training Image:{}".format(train['image'].shape))
print("Size of testing Image:{}".format(test['image'].shape))
''' 
* Save them in directory (save as .np)
'''
# Construct directories first
Dir = os.path.dirname(__file__)
CIFAR_train_Dir = os.path.join(Dir, '..', 'CIFAR', 'train')
CIFAR_test_Dir = os.path.join(Dir, '..', 'CIFAR', 'test')
if not os.path.isdir(CIFAR_train_Dir):
    os.makedirs(CIFAR_train_Dir)
if not os.path.isdir(CIFAR_test_Dir):
    os.makedirs(CIFAR_test_Dir)

# Save images as .png in the directories
for i in range(train['image'].shape[0]):
    print("Save train image:[{}/{}]".format(i +
                                            1, train['image'].shape[0]), end='\r')
    img = Image.fromarray(train['image'][i])
    img.save(os.path.join(CIFAR_train_Dir, '{}.png'.format(i)))
print()
for i in range(test['image'].shape[0]):
    print("Save test image:[{}/{}]".format(i +
                                           1, test['image'].shape[0]), end='\r')
    img = Image.fromarray(test['image'][i])
    img.save(os.path.join(CIFAR_test_Dir, '{}.png'.format(i)))
print()
# Save labels and images as .npy
np.save(os.path.join(CIFAR_train_Dir, 'train_images.npy'), train['image'])
np.save(os.path.join(CIFAR_train_Dir, 'train_labels.npy'), train['label'])
np.save(os.path.join(CIFAR_test_Dir, 'test_images.npy'), test['image'])
np.save(os.path.join(CIFAR_test_Dir, 'test_labels.npy'), test['label'])
