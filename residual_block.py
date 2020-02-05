import torch
import torch.nn as nn
import torch.nn.functional as F


def residual_block(x):
    #squeezing number of channels using 1x1 convolution
    conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
    m = conv1(x)
    m = F.relu(m)

    #3x3 convolution on sqeezed tensor
    conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1, 1))
    m = conv2(m)
    m = F.relu(m)

    #expanding to original number of channels using 1x1 convolution
    conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
    m = conv3(m)
    
    #adding with input tensor
    m = m.add(x)

    m = F.relu(m)

    return m





x = torch.randn(32, #batch
                256,        #channels
                64, 64,  dtype=torch.float)

print (x.shape)

x = residual_block(x)

print (x.shape)

