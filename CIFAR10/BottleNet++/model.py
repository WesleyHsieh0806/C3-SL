import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchsummary import summary
from utils import circular_conv, circular_corr, normalize_for_circular
from thop import profile
from pthflops import count_ops
from ptflops import get_model_complexity_info


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


class SplitResNet50(nn.Module):
    def __init__(self, num_class=10, learning_rate=1e-4, split="linear", compress_ratio=64):
        '''
        * split: The split point
        * compress_ratio: Total compression ratio
        '''
        assert (compress_ratio >= 4) or (split == "middle-2")
        assert ((compress_ratio % 4) == 0) or (split == "middle-2")
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
        # Variables for compression module
        if split == "early":
            spatial = 1
            input_channel = 64
            out_channel = input_channel // (compress_ratio // 4)
        elif split == "middle":
            spatial = 1
            input_channel = 512
            out_channel = input_channel // (compress_ratio // 4)
        elif split == "middle-2" and (compress_ratio == 2):
            spatial = 0
            input_channel = 1024
            out_channel = input_channel // (compress_ratio)
        elif split == "middle-2":
            spatial = 1
            input_channel = 1024
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
        encode_z = self.models[2].encode(z)
        self.front.append(encode_z)
        self.front.append(z)
        '''
        * Cloud
        '''
        # use requires_grad() Since the result of detach never requires grad
        # detach() returns a new tensor but shares the old memory
        # the gradient of tensor z.detach() != the gradient of z
        self.remote.append(encode_z.detach().requires_grad_())
        # Decode
        decrypt_z = self.models[2].decode(self.remote[0])
        self.remote.append(decrypt_z)

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
    model = SplitResNet50(split="middle-2", compress_ratio=2)
    model.train()
    print(summary(model.models[0], (3, 32, 32)))
    print(summary(model.models[2], (1024, 2, 2)))
    macs, params = get_model_complexity_info(model.models[2], (1024, 2, 2), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
