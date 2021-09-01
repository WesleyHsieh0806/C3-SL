import torch
import torchvision.models
import torch.nn as nn
from math import sqrt

from utils import circular_conv, circular_corr, normalize_for_circular


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
        * image->z | Split | z->output
        '''
        self.front = []
        self.remote = []
        z = self.models[0](image)
        z = z.flatten(start_dim=1)
        z, _, _ = normalize_for_circular(z)  # normalize
        self.front.append(z)

        if self.key == None:
            self.key, encrypt_z = circular_conv(z)
            key = self.key
        else:
            key = self.key[:, :, :z.shape[0], :]
            encrypt_z = circular_conv(z, key)

        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        recover_z = circular_corr(encrypt_z, key)
        self.front.append(recover_z)
        self.remote.append(recover_z.detach().requires_grad_())

        return self.models[1](self.remote[0].reshape([len(self.remote[0]), -1]))

    def backward(self, loss):
        ''' When we call loss.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        loss.backward()
        # Copy the gradient
        grad_z = self.remote[0].grad.clone()
        # mathematically, normalize before encrypt and reduce the reconstruction loss
        norm_grad_z, STD, MEAN = normalize_for_circular(
            grad_z)

        key = self.key[:, :, :norm_grad_z.shape[0], :]  # (1,1,B, nofeature)
        encrypt_grad_z = circular_conv(norm_grad_z, key)

        # Decrypt the gradient
        norm_recover_grad_z = circular_corr(encrypt_grad_z, key)

        # Cancel the normalization
        recover_grad_z = norm_recover_grad_z * \
            (STD*sqrt(norm_recover_grad_z.shape[-1]))
        recover_grad_z = recover_grad_z + MEAN

        self.front[0].backward(recover_grad_z)
        # Return the loss between gradient
        return torch.mean((recover_grad_z-grad_z)**2).detach()

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


class SplitResNet50(nn.Module):
    def __init__(self, num_class=10, learning_rate=1e-4):
        super(SplitResNet50, self).__init__()
        # We have to change the last FC layer to output num_class scores
        model = torchvision.models.resnet50(pretrained=True)

        # Split the AlexeNet into two parts: Conv + FC
        self.models = []  # [ConvBlocks, FC]

        model.fc = nn.Linear(2048, num_class)
        layer_list = list(model.children())
        # Convblocks
        self.models.append(
            nn.Sequential(
                *layer_list[:8]
            )
        )
        # FC
        self.models.append(
            nn.Sequential(
                layer_list[8],
                nn.Flatten(start_dim=1),
                layer_list[9],
            )
        )

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
        * image->z | Split | z->output
        '''
        self.front = []
        self.remote = []
        z = self.models[0](image)
        shape = z.shape
        z = z.flatten(start_dim=1)
        z, _, _ = normalize_for_circular(z)  # normalize
        self.front.append(z)

        if self.key == None:
            self.key, encrypt_z = circular_conv(z)
            key = self.key
        else:
            key = self.key[:, :, :z.shape[0], :]
            encrypt_z = circular_conv(z, key)

        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        recover_z = circular_corr(encrypt_z, key)
        self.front.append(recover_z)
        self.remote.append(recover_z.detach().requires_grad_())

        return self.models[1](self.remote[0].reshape(shape))

    def backward(self, loss):
        ''' When we call loss.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        loss.backward()
        # Copy the gradient
        grad_z = self.remote[0].grad.clone()
        # mathematically, normalize before encrypt and reduce the reconstruction loss
        norm_grad_z, STD, MEAN = normalize_for_circular(
            grad_z)

        key = self.key[:, :, :norm_grad_z.shape[0], :]  # (1,1,B, nofeature)
        encrypt_grad_z = circular_conv(norm_grad_z, key)

        # Decrypt the gradient
        norm_recover_grad_z = circular_corr(encrypt_grad_z, key)

        # Cancel the normalization
        recover_grad_z = norm_recover_grad_z * \
            (STD*sqrt(norm_recover_grad_z.shape[-1]))
        recover_grad_z = recover_grad_z + MEAN

        self.front[0].backward(recover_grad_z)
        # Return the loss between gradient
        return torch.mean((recover_grad_z-grad_z)**2).detach()

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
