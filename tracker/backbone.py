import math
import torch
from torch import nn


class AlexNetV1ResBoF24(nn.Module):
    def __init__(self):
        super(AlexNetV1ResBoF24, self).__init__()
        self.base = 'alexnetv1'

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.bof1 = LBoF(input_dim=256, n_codewords=256, kernel_size=5, stride=1, pooling_size=5)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.bof2 = LBoF(input_dim=384, n_codewords=256, kernel_size=3, stride=1, pooling_size=3)
        self.output_dim = self.bof1.n_codewords + self.bof2.n_codewords
        self.initialize_weights()

    def initialize_weights(self, f=1.0):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(f)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        xh1 = self.bof1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        xh2 = self.bof2(x)
        x = torch.cat([xh1, xh2], dim=1)
        return x


class LBoF(nn.Module):
    def __init__(self, input_dim, n_codewords, kernel_size, stride, pooling_size):
        super(LBoF, self).__init__()
        self.conv = nn.Conv2d(input_dim, n_codewords, kernel_size, stride, bias=False)
        self.norm = Normalization()
        self.pool = nn.AvgPool2d(pooling_size, stride=1)
        self.n_codewords = n_codewords

    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.pool(out)
        return out


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()

    def forward(self, input):
        similarities = torch.abs(input)
        out = similarities / (torch.sum(similarities, dim=1, keepdim=True) + 1e-18)
        return out
