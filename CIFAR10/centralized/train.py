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
import argparse

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
* Argument parser
'''
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", required=False, type=int,
                    default=30, help="The Epoch")
parser.add_argument("--dump_path", required=True, type=str,
                    default='./log',
                    help="The saved path of logs and models(Relative)")
args = parser.parse_args()

# create directory for saved_path
saved_path = os.path.join(Dir, args.dump_path)
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


batch_size = 32

Train_Dataset = ImageDataset(train_image, train_labels, train_transform)
Test_Dataset = ImageDataset(test_image, test_labels, test_transform)
Train_Loader = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True)
Test_Loader = DataLoader(Test_Dataset, batch_size=batch_size, shuffle=False)

''' 
* Model Architecture: Alexnet
'''


class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        # We have to change the last FC layer to output num_class scores
        self.model = torchvision.models.alexnet(pretrained=True)
        # Modify the last layer
        self.model.features[0] = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier[6] = nn.Linear(4096, num_class)

    def forward(self, x):
        return self.model(x)


learning_rate = 1e-4
num_epoch = 50

model = AlexNet().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Check the architecture of Alexnet
print("{:=^40}".format("Architecture"))
print(model)
print("{:=^40}".format("End"))

best_acc = 0.
best_loss = float("inf")
train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []

for epoch in range(1, num_epoch+1):
    ''' Training part'''
    model.train()
    train_loss = 0.
    train_acc = 0.
    test_loss = 0.
    test_acc = 0.
    epoch_start_time = time.time()
    for i, (train_x, train_y) in enumerate(Train_Loader, 1):
        print("Batch [{}/{}]".format(i, len(Train_Loader)), end='\r')
        train_x = train_x.cuda()
        train_y = train_y.cuda()

        # y_pred: (batch size, 10)
        y_pred = model(train_x)

        # Compute the loss
        batch_loss = loss(y_pred, train_y)

        # Clean the gradient
        optimizer.zero_grad()

        # Compute the gradient
        batch_loss.backward()

        # Update the model
        optimizer.step()

        train_loss += len(train_x) * batch_loss.item()
        train_acc += np.sum(np.argmax(y_pred.detach().cpu().numpy(),
                                      axis=1) == train_y.cpu().numpy())
    train_loss /= Train_Dataset.__len__()
    train_acc /= Train_Dataset.__len__()

    # Testing part
    model.eval()
    with torch.no_grad():
        for test_x, test_y in Test_Loader:
            test_x = test_x.cuda()
            test_y = test_y.cuda()

            y_pred = model(test_x)

            # Compute the loss and acc
            test_loss += loss(y_pred, test_y).item() * len(test_x)
            test_acc += np.sum(np.argmax(y_pred.detach().cpu().numpy(),
                                         axis=1) == test_y.cpu().numpy())
    test_loss /= len(Test_Dataset)
    test_acc /= len(Test_Dataset)
    # Output the result
    print("Epoch [{}/{}] Time:{:.3f} secs Train_acc:{:.4f} Train_loss:{:.4f}".format(epoch, num_epoch, time.time()-epoch_start_time,
                                                                                     train_acc, train_loss))
    print("Test_acc:{:.4f} Test_loss:{:.4f}".format(
        test_acc, test_loss))

    # Append the accuracy and loss to list
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

    ''' Save the best model '''
    if test_acc > best_acc:
        best_acc = test_acc
        best_loss = test_loss
        print("Save model with Test_acc:{:.4f} Test_loss:{:.4f} at {}".format(
            best_acc, best_loss, os.path.join(saved_path, "Alexnet.pth")))
        torch.save(model.state_dict(), os.path.join(
            saved_path, "Alexnet.pth"))

# Record the train acc and train loss
with open(os.path.join(saved_path, "train_accuracy.csv"), "w") as f:
    for i in range(len(train_acc_list)-1):
        f.write(str(train_acc_list[i])+",")
    f.write(str(train_acc_list[-1]))
with open(os.path.join(saved_path, "train_loss.csv"), "w") as f:
    for i in range(len(train_loss_list)-1):
        f.write(str(train_loss_list[i])+",")
    f.write(str(train_loss_list[-1]))

# Record the validation accuracy and validation loss
with open(os.path.join(saved_path, "test_accuracy.csv"), "w") as f:
    for i in range(len(test_acc_list)-1):
        f.write(str(test_acc_list[i])+",")
    f.write(str(test_acc_list[-1]))

with open(os.path.join(saved_path, "test_loss.csv"), "w") as f:
    for i in range(len(test_loss_list)-1):
        f.write(str(test_loss_list[i])+",")
    f.write(str(test_loss_list[-1]))
