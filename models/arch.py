import torch.nn.functional as F
from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, in_channels = 1):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.build()
    def build(self):
        cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128]
        layers = []
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                self.in_channels = v
        layers += [nn.AdaptiveAvgPool2d(2)]
        self.features = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.bn(x)
        x = torch.utils.checkpoint.checkpoint(self.features, x)
        x = x.view(x.size(0), -1)
        return x

class CNNRegressor(nn.Module):
    """Generic CNN classifier model for ADDA."""
    def __init__(self):
        """Init CNN encoder."""
        super(CNNRegressor, self).__init__()
        self.fc2 = nn.Linear(512, 4)

    def forward(self, feat):
        """Forward the CNN classifier."""
        out = F.dropout(F.relu(feat))
        out = self.fc2(out)
        y = torch.clone(out)
        y[:,2:] = torch.square(out[:,2:])
        return y
        
