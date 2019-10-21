import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizerNet(nn.Module):

    def __init__(self):
        super(RegularizerNet, self).__init__()

        self.eta = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True) #initialized to 0.1

        self.conv_layer_initial = nn.Conv2d(3, 128, kernel_size=1, padding=0, bias=0.001)

        self.residual_block_1 = ResidualBlock()
        self.residual_block_2 = ResidualBlock()

        self.conv_layer_0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=0.001)
        self.conv_layer_1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=0.001)
        self.conv_layer_2 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=0.001)

    def forward(self, x):

# todo: remove "patch mean"

        out = self.conv_layer_initial(x)
        out = F.relu(out)
        out = self.residual_block_2(out)
        out = self.conv_layer_0(out)
        out = F.relu(out)
        out = self.conv_layer_1(out)
        out = F.relu(out)
        out = self.conv_layer_2(out)

#    todo: add "patch mean"
#    todo: add bias according to paper (also to residual block)

        return out


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv_layer_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=0.001)
        self.conv_layer_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=0.001)

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = F.relu(out)
        out = self.conv_layer_2(out)
        out = F.relu(out)
        out = x + out
        return out
