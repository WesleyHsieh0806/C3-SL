import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.models
import torch.nn as nn
from utils import circular_conv, circular_corr


class ECC():
    def __init__(self, nofkey) -> None:

    def generate_key(self, nofkey):
        # Generate the key
        mean = 0
        dim = 1
        for i in range(len(W.shape)):
            dim *= W.shape[i]
        std = (1/dim) ** (1/2)
        Key = (torch.randn(W.shape)*std + mean).cuda()
        Key /= torch.norm(Key, dim=1, keepdim=True)
        Key = Key.reshape([1, 1, Key.shape[0], -1])


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
        * Reconstruction loss is computed between z and recover_z
        '''
        self.front = []
        self.remote = []
        z = self.models[0](image)
        z = z.flatten(start_dim=1)
        self.front.append(z)

        # Encode z(batch, nof_feature) by circulation convolution
        if self.key is None:
            self.key, compress_V = circular_conv(z)
            key = self.key
        else:
            # key(1,1,batch,noffeature)
            key = self.key[:, :, :z.shape[0], :]
            compress_V = circular_conv(z, key)
        # Compress V (1, nof_feature)

        # Decode the compress_V and reshape
        recover_z = circular_corr(compress_V, key)

        # Store recover_z for later reconstruction loss
        self.front.append(recover_z)

        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(recover_z.detach().requires_grad_())

        return self.models[1](self.remote[0])

    def backward(self, L_CE, L_rec, Lambda=0.2):
        ''' When we call L_CE.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        # Compute the total loss first
        L = L_CE + Lambda * L_rec
        L.backward(retain_graph=True)

        ''' 
        * image <-z | KoV=z.grad <- compressV <- K * grad_z=V |<- remote_z.grad <-output
        '''
        # Copy the gradient
        # (batch, nof_feature)
        bs = self.remote[0].shape[0]
        remote_grad_z = self.remote[0].grad.clone()

        # Sum up all the gradients -> (1, nof_featrue)
        remote_grad_z = torch.mean(remote_grad_z, dim=0, keepdim=True)

        key = self.key[:, :, :remote_grad_z.shape[0], :]  # (1,1,B, nofeature)

        # Encode the gradient
        compress_V = circular_conv(remote_grad_z, key)

        # Decode the V
        grad_z = circular_corr(compress_V, key)
        self.front[0].backward(grad_z.repeat([bs, 1]))

        # Return the loss between gradient
        return torch.mean((remote_grad_z-grad_z)**2).detach()

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
