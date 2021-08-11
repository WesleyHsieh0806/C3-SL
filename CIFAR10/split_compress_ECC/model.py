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


class ECC():
    ''' Encryption and Compression Module'''

    def __init__(self) -> None:
        self.key = None

    def generate_key(self, W):
        # * Key:(Batch size, nof_feature)->(1, 1, Batch size, nof_feature)
        mean = 0
        dim = 1
        for i in range(len(W.shape)):
            dim *= W.shape[i]
        std = (1/dim) ** (1/2)
        Key = (torch.randn(W.shape)*std + mean).cuda()
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

        # Decode the compress_V and reshape
        recover_z = circular_corr(compress_V, key)

        return compress_V, recover_z

    def decrypt(self, compress_V):
        key = self.key[:, :, :self.bs, :]
        return circular_corr(compress_V, key)

    def encrypt_Compressed_grad(self, compress_grad):
        key = self.key[:, :, :1, :]
        return circular_conv(compress_grad, key)

    def decrypt_Compressed_grad(self, en_compress_grad):
        key = self.key[:, :, :1, :]
        return circular_corr(en_compress_grad, key)


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
        z = z.flatten(start_dim=1)
        z, _, _ = normalize_for_circular(z)  # normalize

        # ECC Encryption
        compress_V, recover_z = self.ecc(z)
        self.front = [z, recover_z, compress_V]

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

        return self.models[1](remote_recover_z)

    def backward(self, L_CE, L_rec, Lambda=0.2):
        ''' When we call L_CE.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        # Compute the total loss first
        L = L_CE + Lambda * L_rec
        L.backward(retain_graph=True)

        ''' 
        * image <-compress_V.grad <-|<- K o (K* remote_compress_V.grad) <- K * remote_compress_V.grad |<- remote_compress_V.grad <-output
        '''
        # Copy the gradient
        # (1, nof_feature)
        remote_grad_CompressV = self.remote[0].grad.clone()

        # # mathematically, normalize before encrypt and reduce the reconstruction loss
        # norm_remote_grad_CompressV, STD, MEAN = normalize_for_circular(
        #     remote_grad_CompressV)

        # # Encrpt the gradient
        # en_grad_CompressV = self.ecc.encrypt_Compressed_grad(
        #     norm_remote_grad_CompressV)

        # # Decode the V
        # grad_CompressV = self.ecc.decrypt_Compressed_grad(en_grad_CompressV)

        # # Cancel the normalization
        # de_grad_CompressV = grad_CompressV * \
        #     (STD*sqrt(grad_CompressV.shape[-1]))
        # de_grad_CompressV = de_grad_CompressV + MEAN
        de_grad_CompressV = remote_grad_CompressV
        self.front[2].backward(de_grad_CompressV)

        # Return the loss between gradient
        return torch.mean((de_grad_CompressV-remote_grad_CompressV)**2).detach()

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
