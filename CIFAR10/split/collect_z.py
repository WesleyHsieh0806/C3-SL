import pickle
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from argparse import ArgumentParser

from torchvision.transforms.transforms import Lambda

'''
* Reference https://bExponential Lambda Log.openmined.org/split-neural-networks-on-pysyft/
* Corresponding experiments: Training with different batch size
'''
Dir = os.path.dirname(__file__)
train_dir = os.path.join(Dir, '..', 'CIFAR', 'train')
test_dir = os.path.join(Dir, '..', 'CIFAR', 'test')

'''
* Load the data from .npy
'''
train_image = np.load(os.path.join(train_dir, 'train_images.npy'))
train_labels = np.load(os.path.join(train_dir, 'train_labels.npy'))
test_image = np.load(os.path.join(test_dir, 'test_images.npy'))
test_labels = np.load(os.path.join(test_dir, 'test_labels.npy'))

print("Size of training images:{}".format(train_image.shape))
print("Size of training labels:{}".format(train_labels.shape))
print("Size of testing images:{}".format(test_image.shape))
print("Size of testing labels:{}".format(test_labels.shape))

'''
* Argument Parser
'''
parser = ArgumentParser()
parser.add_argument("--batch", required=False, type=int,
                    default=32, help="The batch size")
parser.add_argument("--epoch", required=False, type=int,
                    default=30, help="The batch size")
parser.add_argument("--Lambda", required=False, type=float,
                    default=0.0001, help="The batch size")
parser.add_argument("--dump_path", required=False, type=str,
                    default='./logs', help="The directory to save logs and models")
parser.add_argument("--restore_path", required=False, type=str,
                    default='./logs/Lambda_0_Batch64', help="The directory of pickle to restore the model")
args = parser.parse_args()

# Create directory for dump path
saved_path = os.path.join(
    os.path.dirname(__file__), args.dump_path)
if not os.path.isdir(saved_path):
    os.makedirs(saved_path)
'''
********************************************
* Data Augmentation and Dataset, DataLoader
********************************************
'''
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize([128, 128]),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize([128, 128]),
    transforms.ToTensor(),
])


class ImageDataset(Dataset):
    def __init__(self, x, y, transform):
        '''
        * x: training data
        * y: training label
        * transform: transforms that will be operated on images
        '''
        self.x = x
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        '''
        * The Data loader will create integers which indicates indices
        * And the Dataset should return the samples which mapped by these indices
        '''
        X = self.x[index]
        Y = self.y[index]

        # Transform X into tensor( Y is already long tensor)
        X = self.transform(X)
        return X, Y


batch_size = args.batch

Train_Dataset = ImageDataset(train_image, train_labels, train_transform)
Test_Dataset = ImageDataset(test_image, test_labels, test_transform)
Train_Loader = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True)
Test_Loader = DataLoader(Test_Dataset, batch_size=batch_size, shuffle=False)

'''
* Model Architecture: Alexnet
'''


learning_rate = 1e-4
Lambdas = [args.Lambda*i/args.epoch for i in range(args.epoch+1)]
num_epoch = args.epoch

if args.restore_path:
    saved_dict = torch.load(os.path.join(args.restore_path, "./Alexnet.pth"))
    print(saved_dict.keys())
    model = saved_dict["Model"]
model.cuda()
# model.load_state_dict(torch.load(
#     os.path.join(args.restore_path, "Alexnet.pth")), strict=True)
CE_Loss = nn.CrossEntropyLoss()

# Check the architecture of Alexnet
print("{:=^40}".format("Architecture"))
print(model.models[0])
print(model.models[1])
print(model.state_dict())
print("{:=^40}".format("End"))

best_acc = 0.
best_loss = float("inf")

for epoch in range(1):
    ''' Training part'''
    Lambda = Lambdas[epoch]
    model.train()
    test_acc = 0.
    epoch_start_time = time.time()

    z = []
    recover_z = []
    with torch.no_grad():
        for i, (train_x, train_y) in enumerate(Train_Loader, 1):
            print("Batch [{}/{}]".format(i, len(Train_Loader)), end='\r')
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            # y_pred: (batch size, 10)
            y_pred = model(train_x)

            # Compute the loss
            batch_L_CE = CE_Loss(y_pred, train_y)
            batch_L_rec = torch.mean((model.front[1]-model.front[0])**2)
            z.append(model.front[0].detach().cpu().numpy())
            recover_z.append(model.front[1].detach().cpu().numpy())
    z = np.concatenate(z)
    recover_z = np.concatenate(recover_z)

    # Testing part
    model.eval()
    with torch.no_grad():
        for test_x, test_y in Test_Loader:
            test_x = test_x.cuda()
            test_y = test_y.cuda()

            y_pred = model(test_x)

            # Compute the loss and acc
            test_acc += np.sum(np.argmax(y_pred.detach().cpu().numpy(),
                                         axis=1) == test_y.cpu().numpy())
    test_acc /= len(Test_Dataset)
    # Output the result
    print("Epoch [{}/{}] Time:{:.3f} secs".format(epoch,
                                                  num_epoch, time.time()-epoch_start_time))
    print("Test_acc:{:.4f} ".format(
        test_acc))
stored_dict = {
    "z": z,
    "recover_z": recover_z
}
with open(os.path.join(args.dump_path, "features.pkl"), 'wb') as f:
    pickle.dump(stored_dict, f)
