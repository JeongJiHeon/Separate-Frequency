import torch
import torch.nn as nn
from octconv import *

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, 
                      stride = 1, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, 
                      stride = 1, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x):
        return self.block(x) + x
    def __name__(self):
        return 'ResidualBlock'
    



class Generator(nn.Module):
    def __init__(self, n = 9):
        
        """
        ----------------------------------------------------------------------------------------
        
        Set Parameter:
            n           : number of residual block                           ( default : 9     )
                - Recommend n = 9 for image size = 256 x 256
                            n = 6 for image size = 128 x 128
            
        ----------------------------------------------------------------------------------------
        """
        super(Generator, self).__init__()
        dim = 32
        self.block1 = self._make_Convblock(3, dim, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect')

        self.block2 = nn.Sequential(
            self._make_Convblock(dim, dim*2),
            self._make_Convblock(dim*2, dim*4)
        )


        self.block3 = nn.Sequential( *[ ResidualBlock(dim*4) for n in range(n) ])

        self.block4 = nn.Sequential(
            self._make_ConvTblock(dim*4, dim*2, kernel_size = 4),
            self._make_ConvTblock(dim*2, dim, kernel_size = 4)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = 3, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect'),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


    def __name__(self):
        return 'Generator'




    def _make_Convblock(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True, 
                        padding_mode = 'zero', norm = nn.InstanceNorm2d, act = nn.ReLU(inplace = False)):
        
        block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                      kernel_size = kernel_size, stride = stride, padding = padding,
                      bias = bias, padding_mode = padding_mode),
            norm(out_channels),
            act
        )
        return block

    def _make_ConvTblock(self, in_channels, out_channels, kernel_size=3, stride=2, 
                         padding=1, bias=True, norm = nn.InstanceNorm2d, act = nn.ReLU):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = kernel_size, stride = stride, padding = padding, bias = bias),
            norm(out_channels),
            act()
        )
        return block


    
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self, a = 0.25):
        
        """
        ----------------------------------------------------------------------------------------
        
        Set Parameter:
            a           : Octave Coefficient                                 ( default : 1/8   )
            
        ----------------------------------------------------------------------------------------
        """        
        super(Discriminator, self).__init__()
        self.a = a
        dim = 32
        self.block1 = FirstOctaveR(in_channels = 3, out_channels = dim, 
                                   kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True)
        self.block2 = OctaveCNR(in_channels = dim, out_channels = dim*2, 
                                kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)
        self.block3 = OctaveCNR(in_channels = dim*2, out_channels = dim*4, 
                                kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)
        self.block4 = OctaveCNR(in_channels = dim*4, out_channels = dim*8, 
                                kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)


        self.s_block1 = nn.Conv2d(int(dim*8*(1-a)), 1, kernel_size = 1, stride = 1, padding = 0)
        self.s_block2 = nn.Conv2d(int(dim*8*  a  ), 1, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)


        x_h = nn.Sigmoid()(self.s_block1(x[0]).squeeze(1))
        x_l = nn.Sigmoid()(self.s_block2(x[1]).squeeze(1))

        return x_h, x_l
    def __name__(self):
        return 'Octave{:.3f} Discriminator'.format(self.a)






    
    
    
    
