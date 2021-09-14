import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.models
import torch.nn as nn
from utils import circular_conv, circular_corr, normalize_for_circular
from math import sqrt
from torchsummary import summary
import torch.nn.functional as F


class compression_module(nn.Module):
    def __init__(self, input_channel=2048, hidden_channel=32, channel=1, spatial=0):
        super(compression_module, self).__init__()

        # Spatial=1 means spatial compression, while 0 means no spatial compression
        self.conv1 = nn.Conv2d(input_channel, hidden_channel,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channel, input_channel,
                               kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(hidden_channel)
        self.batchnorm2 = nn.BatchNorm2d(input_channel)

        self.conv3 = nn.Conv2d(
            input_channel, hidden_channel, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(
            hidden_channel, input_channel, kernel_size=2, stride=2)

        self.channel = channel
        self.spatial = spatial

    def encode(self, x):
        H = x.size()[2]

        C = x.size()[1]

        B = x.size()[0]

        if self.spatial == 0:
            x = torch.sigmoid(self.batchnorm1(self.conv1(x)))

        elif self.spatial == 1:
            x = torch.sigmoid(self.batchnorm1(self.conv3(x)))
        return x

    def decode(self, x):
        if self.spatial == 0:
            x = F.relu(self.batchnorm2(self.conv2(x)))
        else:
            x = F.relu(self.batchnorm2(self.conv4(x)))
        return x

    def forward(self, x):
        H = x.size()[2]

        C = x.size()[1]

        B = x.size()[0]

        if self.spatial == 0:
            x = torch.sigmoid(self.batchnorm1(self.conv1(x)))

        elif self.spatial == 1:
            x = torch.sigmoid(self.batchnorm1(self.conv3(x)))
        if self.spatial == 0:
            x = F.relu(self.batchnorm2(self.conv2(x)))
        else:
            x = F.relu(self.batchnorm2(self.conv4(x)))
        return x


class ECC():
    ''' Encryption and Compression Module'''

    def __init__(self) -> None:
        self.key = None

    def generate_key(self, W):
        # * Key:(Batch size, nof_feature)->(1, 1, Batch size, nof_feature)
        mean = 0
        dim = 1
        for i in range(1, len(W.shape)):
            dim *= W.shape[i]
        std = (1/dim) ** (1/2)
        Key = (torch.randn(W.shape)*std + mean)
        Key /= torch.norm(Key, dim=1, keepdim=True)
        Key = Key.reshape([1, 1, Key.shape[0], -1])
        self.key = Key

    def __call__(self, z):
        ''' 
        * Return: Compressed_V(Detached) and  recover_z
        '''
        # z:(Batch, noffeature)

        # Encode z(batch, nof_feature) by circulation convolution
        if self.key is None:
            self.generate_key(z)

        # The last batch normally contains less samples, so we have to adjust the key here
        self.bs = z.shape[0]
        key = self.key[:, :, :self.bs, :]

        compress_V = circular_conv(z, key)
        # Compress V (1, nof_feature)

        return compress_V

    def decrypt(self, compress_V):
        key = self.key[:, :, :self.bs, :]
        return circular_corr(compress_V, key)

    def encrypt_Compressed_grad(self, compress_grad):
        key = self.key[:, :, :1, :]
        return circular_conv(compress_grad, key)

    def decrypt_Compressed_grad(self, en_compress_grad):
        key = self.key[:, :, :1, :]
        return circular_corr(en_compress_grad, key)


class SplitResNet50(nn.Module):
    def __init__(self, num_class=10, learning_rate=1e-4, split="linear", compress_ratio=64):
        '''
        * split: The split point
        * compress_ratio: Total compression ratio
        '''
        assert compress_ratio >= 4
        assert (compress_ratio % 4) == 0
        assert split in ["linear", "early", "middle"]

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
            "linear": 8
        }
        self.split = split
        # Variables for compression module
        if split == "early":
            spatial = 1
            input_channel = 64
            out_channel = input_channel // (compress_ratio // 4)
        elif split == "middle":
            spatial = 1
            input_channel = 512
            out_channel = input_channel // (compress_ratio // 4)
        elif split == "linear":
            # No spatial compression
            spatial = 0
            input_channel = 2048
            out_channel = input_channel // (compress_ratio)

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

        # Compression module
        self.models.append(compression_module(
            input_channel=input_channel, hidden_channel=out_channel, spatial=spatial))

        # Two optimizers for each half
        self.optimizers = [torch.optim.Adam(
            model.parameters(), lr=learning_rate) for model in self.models]
        # Encryption and Compression Module
        self.ecc = ECC()

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

        # Encode
        encode_z = self.models[2].encode(z)

        shape = encode_z.shape
        encode_z = encode_z.flatten(start_dim=1)
        encode_z, STD, MEAN = normalize_for_circular(encode_z)  # normalize

        # ECC Encryption
        compress_V = self.ecc(encode_z)
        self.front = [z, compress_V]

        ''' ******
        * Cloud  *
        **********
        '''

        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(compress_V.detach().requires_grad_())

        # ECC Decryption
        remote_en_recover_z = self.ecc.decrypt(self.remote[0])

        # Decode
        remote_en_recover_z = remote_en_recover_z.reshape(shape)
        remote_de_recover_z = self.models[2].decode(remote_en_recover_z)
        self.remote.append(remote_de_recover_z)

        return self.models[1](remote_de_recover_z)

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
        self.front[1].backward(grad_CompressV)

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
    input = torch.zeros([2, 3, 32, 32])
    model = SplitResNet50(split="middle", compress_ratio=64)
    print(model(input))
