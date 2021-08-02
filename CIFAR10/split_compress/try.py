import torch
import torch.nn.functional as F
from utils import circular_corr, circular_conv, circular_pad2d
import time
import torchvision
import torch.nn as nn
from torchsummary import summary
# W = torch.tensor([[1., 2], [3., 4]]).reshape([1, 1, 2,  2])
# W = torch.randn(1, 1, 32, 9216)
# Key = torch.randn(1, 1, 32, 9216)
# Key, V = circular_conv(W)
# compress_V = torch.sum(V, dim=0, keepdims=True)
# Recover_W = circular_corr(compress_V, Key)
# print(W.shape)
# print(Recover_W.shape)


class SplitAlexNet(nn.Module):
    def __init__(self, cut_layer, num_class=10, learning_rate=1e-4):
        '''
        * cutlayer: the number of conv blocks to be cut at
        * cutlayer=1 denotes we split AlexNet at the output of first conv blocks
        '''
        super(SplitAlexNet, self).__init__()
        # We have to change the last FC layer to output num_class scores
        model = torchvision.models.alexnet(pretrained=True)
        # Split the AlexeNet into two parts: Conv + (Conv+FC)
        self.models = []  # [ConvBlocks, ConvBlocks+FC]

        block_idx = {1: 2,
                     2: 5, 3: 7, 4: 9, 5: 12}  # the first conv blocks stops at idx2, the second conv blocks stop at idx 5
        idx = block_idx[cut_layer]+1
        layer_list = list(model.children())
        # Convblocks
        model.features[0] = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(4096, num_class)
        self.models.append(layer_list[0][:idx])

        # FC
        self.models.append(
            nn.Sequential(layer_list[0][idx:],
                          layer_list[1],
                          nn.Flatten(start_dim=1),
                          layer_list[2],
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
        * image->z | z-> K*z=V -> compressV -> K o compressV |-> z->output
        '''
        self.front = []
        self.remote = []
        z = self.models[0](image)
        shape = z.shape
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
        return self.models[1](self.remote[0].reshape(shape))

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


model = SplitAlexNet(5)
print(summary(model.models[0], (3, 32, 32)))
# print(summary(model.models[1], (3, 32, 32)))
