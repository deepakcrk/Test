import torch
import torch.nn as nn
import torch.nn.functional as F

#inverted residual with linear bottleneck
def inverted_linear_residual_block(x):
    #expanding number of channels using 1x1 convolution
    conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
    m = conv1(x)
    m = F.relu(m)

    #3x3 depthwise convolution, on the expanded tensor
    conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1, 1), groups=512)
    m = conv2(m)
    m = F.relu(m)

    #squeezing to original number of channels using 1x1 convolution
    conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
    m = conv3(m)
    
    #adding with input tensor
    m = m.add(x)

    #------------------------------------------------------------------------------------------------------#
    # No non-linear activations like RELU here.                                                            #
    # Non linear transformation is used to preserve information,                                           #
    # Non linearities like ReLU is not used as it can destroy data or cause lose of information            #
    #                                                                                                      #
    # the main difference between 'inverted_residual_block' and 'inverted_linear_residual_block' is that   #
    # there is no non-linear activations in the end for 'inverted_linear_residual_block'                   #
    #------------------------------------------------------------------------------------------------------#


    return m





x = torch.randn(32, #batch
                256,        #channels
                64, 64,  dtype=torch.float)

print (x.shape)

x = inverted_linear_residual_block(x)

print (x.shape)

