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
from utils import circular_conv, circular_corr
''' 
* Reference https://blog.openmined.org/split-neural-networks-on-pysyft/
* Corresponding experiments: Get the Average for each dimension
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
                    default=64, help="The batch size")
parser.add_argument("--epoch", required=False, type=int,
                    default=5, help="The number of epoch")
args = parser.parse_args()
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


class SplitAlexNet(nn.Module):
    def __init__(self, num_class=10, learning_rate=1e-4):
        super(SplitAlexNet, self).__init__()
        # We have to change the last FC layer to output num_class scores
        model = torchvision.models.alexnet(pretrained=True)

        # Split the AlexeNet into two parts: Conv + FC
        self.models = []  # [ConvBlocks, FC]

        # Convblocks
        model.features[0] = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.models.append(nn.Sequential(*list(model.children())[:-1]))

        # FC
        self.models.append(list(model.children())[-1])
        self.models[1][6] = nn.Linear(4096, num_class)

        # Two optimizers for each half
        self.optimizers = [torch.optim.Adam(
            model.parameters(), lr=learning_rate) for model in self.models]
        self.key = None

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def forward(self, image):
        ''' 
        * Notice that we have to store the output of the 
        * front-end model so that we can compute gradient later
        * front = [z] remote = [z.detach().requires_grad()]
        * image->z | z-> K*z=V -> compressV -> K o compressV |-> z->output
        '''
        self.front = []
        self.remote = []
        z = self.models[0](image)
        z = z.flatten(start_dim=1)
        self.front.append(z)
        # Encode z(batch, nof_feature) by circulation convolution
        if self.key is None:
            self.key, compress_V = circular_conv(z.detach())
            key = self.key
        else:
            # key(1,1,batch,noffeature)
            key = self.key[:, :, :z.shape[0], :]
            compress_V = circular_conv(z.detach(), key)
        # Compress V (1, nof_feature)

        # Decode the compress_V and reshape
        recover_z = circular_corr(compress_V, key)

        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(recover_z.requires_grad_())

        return self.models[1](self.remote[0].reshape([len(self.remote[0]), -1])), z

    def backward(self, loss):
        ''' When we call loss.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        loss.backward()

        ''' 
        * image <-z | KoV=z.grad <- compressV <- K * grad_z=V |<- remote_z.grad <-output
        '''
        # Copy the gradient
        # (batch, nof_feature)
        remote_grad_z = self.remote[0].grad.clone()
        key = self.key[:, :, :remote_grad_z.shape[0], :]  # (1,1,B, nofeature)

        # Encode the gradient
        compress_V = circular_conv(remote_grad_z, key)

        # Decode the V
        grad_z = circular_corr(compress_V, key)
        self.front[0].backward(grad_z)
        return grad_z

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        ''' Update parameters for both half models'''
        for opt in self.optimizers:
            opt.step()

    def cuda(self):
        for model in self.models:
            model.cuda()


learning_rate = 1e-4
num_epoch = args.epoch

model = SplitAlexNet()
model.cuda()
# model.load_state_dict(torch.load(os.path.join(
#     Dir, "Alexnet_init.pth")))
loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Check the architecture of Alexnet
print("{:=^40}".format("Architecture"))
print(model.models[0])
print(model.models[1])
print("{:=^40}".format("End"))

best_acc = 0.
best_loss = float("inf")
best_features = []  # Collect the features for each epoch
best_gradients = []  # Collect the gradient for each epoch
best_labels = []  # Collect the labels for each epoch
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
    features = []  # Collect the features for each epoch
    gradients = []  # Collect the gradient for each epoch
    labels = []  # Collect the labels for each epoch
    for i, (train_x, train_y) in enumerate(Train_Loader, 1):
        print("Batch [{}/{}]".format(i, len(Train_Loader)), end='\r')
        train_x = train_x.cuda()
        train_y = train_y.cuda()

        # y_pred: (batch size, 10)
        y_pred, feature = model(train_x)

        # Collect the features (batch, dim)
        features.append(feature.detach().cpu().numpy())

        # Compute the loss
        batch_loss = loss(y_pred, train_y)

        # Clean the gradient
        model.zero_grad()

        # Compute the gradient
        gradient = model.backward(batch_loss)
        gradients.append(gradient.detach().cpu().numpy())
        labels.append(train_y.detach().cpu().numpy())

        # Update the model
        model.step()

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

            y_pred, _ = model(test_x)

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
        best_features = features
        best_gradients = gradients
        best_labels = labels

        print("Save model with Test_acc:{:.4f} Test_loss:{:.4f} at {}".format(
            best_acc, best_loss, os.path.join(
                Dir, "Alexnet_init.pth")))
        torch.save(model.state_dict(), os.path.join(
            Dir, "Alexnet_init.pth"))
if not os.path.isdir(os.path.join(
        os.path.dirname(__file__), "Average")):
    os.makedirs(os.path.join(
        os.path.dirname(__file__), "Average"))

labels = np.concatenate(best_labels)
features = np.concatenate(best_features)
gradients = np.concatenate(best_gradients)

print("Size of labels:{}".format(labels.shape))
print("Size of features:{}".format(features.shape))
print("Size of gradients:{}".format(gradients.shape))


# Compute the mask for each label
nof_label = np.max(labels)+1
start = np.min(labels)
f_avg_bylabel = []  # the Average of the features of each label
grad_avg_bylabel = []  # the Average of the gradients of each label
for label in range(start, nof_label):
    # Label mask
    label_mask = (labels == label)
    f_avg_bylabel.append(np.mean(
        features[label_mask], axis=0, keepdims=True))
    grad_avg_bylabel.append(
        np.mean(gradients[label_mask], axis=0, keepdims=True))
f_avg_bylabel = np.concatenate(f_avg_bylabel, axis=0)
grad_avg_bylabel = np.concatenate(grad_avg_bylabel, axis=0)

print("Size of avg of features:{}".format(f_avg_bylabel.shape))
print("Size of avg of gradients:{}".format(grad_avg_bylabel.shape))

# Save the labels features gradients as np array
np.save(os.path.join(
        os.path.dirname(__file__), "Average", 'features_avg_Batch{}.npy'.format(args.batch)), f_avg_bylabel)
np.save(os.path.join(
        os.path.dirname(__file__), "Average", 'gradients_avg_Batch{}.npy'.format(args.batch)), grad_avg_bylabel)
