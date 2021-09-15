import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.models
import torch.nn as nn
from utils import circular_conv, circular_corr, normalize_for_circular
from math import sqrt, ceil
from torchsummary import summary


class ECC():
    ''' Encryption and Compression Module'''

    def __init__(self, compress_ratio) -> None:
        self.key = None
        self.ratio = compress_ratio

    def generate_key(self, W):
        # * Key:(compress_ratio, nof_feature)->(1, 1, compress_ratio, nof_feature)
        mean = 0
        dim = 1
        for i in range(1, len(W.shape)):
            dim *= W.shape[i]
        std = (1/dim) ** (1/2)

        # Create the key
        keyshape = [W.shape[i] for i in range(len(W.shape))]
        keyshape[0] = self.ratio

        Key = (torch.randn(keyshape)*std + mean)
        if torch.cuda.is_available():
            Key = Key.cuda()
        Key /= torch.norm(Key, dim=1, keepdim=True)
        Key = Key.reshape([1, 1, Key.shape[0], -1])
        self.key = Key

    def __call__(self, z):
        ''' 
        * Return: a tensor that contains all compressV (n_group, n_feature)
        * [Compress_V_1, Compress_V_2, ..., COmpress_V_Batchsize]
        '''
        # z:(Batch, noffeature)

        # Encode z(batch, nof_feature) by circulation convolution
        if self.key is None:
            self.generate_key(z)

        compress_V_list = []
        self.bs = z.shape[0]
        # There are total self.bs//self.ratio groups to be compressed
        n_group = self.bs//self.ratio

        for i in range(n_group):

            group_z = z[i * self.ratio: (i+1) * self.ratio]
            key = self.key[:, :, :group_z.shape[0], :]
            # Compress V (1, nof_feature)
            compress_V = circular_conv(group_z, key)

            # add compress v into list
            compress_V_list.append(compress_V)
        return torch.cat(compress_V_list, dim=0)

    def decrypt(self, compress_V_list):
        # There are total self.bs//self.ratio groups to be compressed
        n_group = self.bs//self.ratio
        recover_z_list = []

        for i in range(n_group):
            compress_V = compress_V_list[i:i+1]
            key = self.key[:, :, :self.ratio, :]
            recover_group_z = circular_corr(compress_V, key)

            recover_z_list.append(recover_group_z)

        return torch.cat(recover_z_list, dim=0)


class SplitResNet50(nn.Module):
    def __init__(self, num_class=10, learning_rate=1e-4, split="linear", compress_ratio=64):
        '''
        * split: The split point
        * compress_ratio: Total compression ratio
        '''
        assert split in ["linear", "early", "middle", "middle-2"]

        super(SplitResNet50, self).__init__()
        # We have to change the last FC layer to output num_class scores
        model = torchvision.models.resnet50(pretrained=True)

        # Split the AlexeNet into two parts: Conv + FC
        self.models = []  # [ConvBlocks, FC]

        model.fc = nn.Linear(2048, num_class)
        layer_list = list(model.children())

        # Split point
        spl_pnt = {
            "early": 4,
            "middle": 6,
            "middle-2": 7,
            "linear": 8
        }
        self.split = split

        # Convblocks
        self.models.append(
            nn.Sequential(
                *layer_list[:spl_pnt[split]]
            )
        )
        # FC
        self.models.append(
            nn.Sequential(
                *layer_list[spl_pnt[split]:9],
                nn.Flatten(start_dim=1),
                layer_list[9],
            )
        )

        # Two optimizers for each half
        self.optimizers = [torch.optim.Adam(
            model.parameters(), lr=learning_rate) for model in self.models]
        # Encryption and Compression Module
        self.ecc = ECC(compress_ratio)

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
        shape = z.shape
        z = z.flatten(start_dim=1)
        if self.split == "linear":
            z, STD, MEAN = normalize_for_circular(z)  # normalize

        # ECC Encryption
        compress_V = self.ecc(z)
        self.front = [compress_V]

        ''' ******
        * Cloud  *
        **********
        '''
        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(compress_V.detach().requires_grad_())

        # ECC Decryption
        remote_recover_z = self.ecc.decrypt(self.remote[0])

        return self.models[1](remote_recover_z.reshape(shape))

    def backward(self, L_CE):
        ''' When we call L_CE.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        # Compute the total loss first
        L = L_CE
        L.backward(retain_graph=True)

        # Copy the gradient
        # (1, nof_feature)
        remote_grad_CompressV = self.remote[0].grad.clone()

        # Send to the edge
        grad_CompressV = remote_grad_CompressV
        self.front[0].backward(grad_CompressV)

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


if __name__ == "__main__":
    model = SplitResNet50(split="middle", compress_ratio=1)
    input = torch.zeros([64, 3, 32, 32])
    model(input)
