import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchsummary import summary
from utils import circular_conv, circular_corr, normalize_for_circular
from thop import profile
from pthflops import count_ops


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

        # Encode
        encrypt_z = z
        self.front.append(encrypt_z)

        '''
        # Cloud
        '''
        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(encrypt_z.detach().requires_grad_())

        # Decode
        decrypt_z = self.remote[0]
        self.front.append(decrypt_z)

        return self.models[1](decrypt_z.flatten(start_dim=1))

    def backward(self, loss):
        ''' When we call loss.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        loss.backward()
        # Copy the gradient
        grad_z = self.remote[0].grad.clone()

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
                *layer_list[:4]
            )
        )
        # FC
        self.models.append(
            nn.Sequential(
                *layer_list[4:8],
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

        # Encode
        encrypt_z = z
        self.front.append(encrypt_z)
        '''
        * Cloud
        '''
        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(encrypt_z.detach().requires_grad_())

        # Decode
        decrypt_z = self.remote[0]
        self.front.append(decrypt_z)

        return self.models[1](decrypt_z)

    def backward(self, loss):
        ''' When we call loss.backward(), it only backwards for the last half layers
        * So here we will manually compute the gradient for the front-half convblocks
        '''
        loss.backward()
        # Copy the gradient
        grad_z = self.remote[0].grad.clone()

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


if __name__ == "__main__":
    model = compression_module(
        input_channel=256, hidden_channel=16, spatial=1)
    print(summary(model, (256, 8, 8)))
    input = torch.zeros([1, 256, 8, 8])
    flop, param = profile(model, inputs=(input, ))
    # Notice that the flop here does not sum up the addition part after multiplication, so the result should be multiplied by 2
    print(flop)
    print(param)
