import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.autograd as autograd

from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from random import *

from torchvision import models
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from model import *
from utils import *
from octconv import *





class Generator(nn.Module):
    def __init__(self, a = 0.25, n = 9):
        
        """
        ----------------------------------------------------------------------------------------
        
        Set Parameter:
            a           : Octave Coefficient                                 ( default : 1/8   )
            n           : number of residual block                           ( default : 9     )
            
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



        #self.block5 = self._make_Convblock(dim, 3, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect', act = nn.Sigmoid())

    def __name__(self):
        return 'Generator'




    def _make_Convblock(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True, padding_mode = 'zero', norm = nn.InstanceNorm2d, act = nn.ReLU(inplace = False)):
        block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding,
                      bias = bias, padding_mode = padding_mode),
            norm(out_channels),
            act
        )
        return block

    def _make_ConvTblock(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True, norm = nn.InstanceNorm2d, act = nn.ReLU):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding,
                      bias = bias),
            norm(out_channels),
            act()
        )
        return block
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x):
        return self.block(x) + x
    def __name__(self):
        return 'ResidualBlock'
    
    
    
    
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
        self.block1 = FirstOctaveR(in_channels = 3, out_channels = dim, kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True)
        self.block2 = OctaveCBR(in_channels = dim, out_channels = dim*2, kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)
        self.block3 = OctaveCBR(in_channels = dim*2, out_channels = dim*4, kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)
        self.block4 = OctaveCBR(in_channels = dim*4, out_channels = dim*8, kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)
    #    self.block5 = OctaveCBR(in_channels = dim*8, out_channels = dim*8, kernel_size = 4, stride = 2, padding = 1, alpha = a, bias = True, norm_layer = nn.InstanceNorm2d)

        self.s_block1 = nn.Conv2d(int(dim*8*(1-a)), 1, kernel_size = 1, stride = 1, padding = 0)
        self.s_block2 = nn.Conv2d(int(dim*8*  a  ), 1, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #x = self.block5(x)


        x_h = nn.Sigmoid()(self.s_block1(x[0]).squeeze(1))
        x_l = nn.Sigmoid()(self.s_block2(x[1]).squeeze(1))

        return x_h, x_l
    def __name__(self):
        return 'Octave{:.3f} Discriminator'.format(self.a)








# Octave Model 기본모델
class CycleGAN():
    def __init__(self, epoch = 100, batch_size = 4, a = 0.125, lr = 0.0002, cycleLambda = 10, random = True, image_size = 256, path = 'output/', loss = 'MSE'):
        
        """
        ----------------------------------------------------------------------------------------
        
        Set Parameter:
            a           : Octave Coefficient                                 ( default : 1/8   )
            lr          : Learning Rate                                      ( default : 0.0002)
            cycleLambda : Cycle Coefficient                                  ( default : 10    )
            random      : Internally Dividing Between High Loss and Low Loss ( default : 1/2   )
            loss        : vanila GAN Loss(BCE) / LSGAN Loss(MSE)             ( default : BCE   )
            
            
        This is a model for GPU User. If you want CPU, Please delete '.cuda()' and use it

        ----------------------------------------------------------------------------------------
        """        
        
        self.dataloaderA = torch.utils.data.DataLoader(dataset('data/monet2photo/trainA', image_size = image_size), shuffle = True, batch_size = batch_size, num_workers = 2)
        self.dataloaderB = torch.utils.data.DataLoader(dataset('data/monet2photo/trainB', image_size = image_size), shuffle = True, batch_size = batch_size, num_workers = 2)
        
        ## For measuring FID score ##
        #self.testloaderA = torch.utils.data.DataLoader(dataset('data/monet2photo/testA', image_size = image_size), batch_size = 10, drop_last = True)
        #self.testloaderB = torch.utils.data.DataLoader(dataset('data/monet2photo/testB', image_size = image_size), batch_size = 10, drop_last = True)

        self.image_size = image_size
        self.lr = lr

        self.fixdataB = next(iter(torch.utils.data.DataLoader(dataset('data/monet2photo/testB', image_size = image_size), batch_size = 25))).cuda()

        self.GeneratorA = Generator(n = 9).cuda() # make B (input A)
        self.GeneratorB = Generator(n = 9).cuda() # make A (input B)

        self.DiscriminatorA = Discriminator(a = a).cuda() # input A
        self.DiscriminatorB = Discriminator(a = a).cuda() # input B

        self.optimGA = torch.optim.Adam(self.GeneratorA.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.optimGB = torch.optim.Adam(self.GeneratorB.parameters(), lr = self.lr, betas = (0.5, 0.999))

        self.optimDA = torch.optim.Adam(self.DiscriminatorA.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.optimDB = torch.optim.Adam(self.DiscriminatorB.parameters(), lr = self.lr, betas = (0.5, 0.999))

        self.cycleloss = nn.L1Loss()
        self.cycleLambda = cycleLambda

        if loss == 'BCE' :
            self.criterion = nn.BCELoss()
        elif loss == 'MSE' :
            self.criterion = nn.MSELoss()
        self.batch_size = batch_size

        self.real_h = torch.ones(batch_size, self.image_size//16, self.image_size//16).cuda()
        self.real_l = torch.ones(batch_size, self.image_size//32, self.image_size//32).cuda()

        self.fake_h = torch.zeros(batch_size, self.image_size//16, self.image_size//16).cuda()
        self.fake_l = torch.zeros(batch_size, self.image_size//32, self.image_size//32).cuda()

        self.random = random

        self.Epoch = epoch

        self.ImagepoolA = []
        self.ImagepoolB = []
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)


    def save(self):
        torch.save(self.DiscriminatorA.state_dict(), self.path + 'DiscriminatorA.pkl')
        torch.save(self.DiscriminatorB.state_dict(), self.path + 'DiscriminatorB.pkl')
        torch.save(self.GeneratorA.state_dict(), self.path + 'GeneratorA.pkl')
        torch.save(self.GeneratorB.state_dict(), self.path + 'GeneratorB.pkl')


    def load(self):

        self.GeneratorA.load_state_dict(torch.load(self.path + 'GeneratorA.pkl'))
        self.GeneratorB.load_state_dict(torch.load(self.path + 'GeneratorB.pkl'))
        self.DiscriminatorA.load_state_dict(torch.load(self.path + 'DiscriminatorA.pkl'))
        self.DiscriminatorB.load_state_dict(torch.load(self.path + 'DiscriminatorB.pkl'))


        self.GeneratorA.train()
        self.GeneratorB.train()
        self.DiscriminatorA.train()
        self.DiscriminatorB.train()


    def set_lr(self, epoch):
        # 
        # learning rate decreases linearly to zero if epoch > EPOCH//2
        #
        self.optimGA.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)
        self.optimGB.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)
        self.optimDA.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)
        self.optimDB.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)



    def train(self, step = 0):
        saveimage(self.fixdataB, self.Epoch, path = self.path)
        for epoch in range(self.Epoch):
            if step > epoch:
                continue

            saveimage(self.GeneratorB(self.fixdataB), epoch, path = self.path)
            self.set_lr(epoch)




            pbar = tqdm(enumerate(zip(self.dataloaderA, self.dataloaderB)), total = len(self.dataloaderA))

            for idx,(imgA, imgB) in pbar:

                p = random() if self.random else 0.5


                imgA = imgA.cuda()
                imgB = imgB.cuda()

                # A->B
                outputB = self.GeneratorA(imgA)
                
                # B->A
                outputA = self.GeneratorB(imgB)
                
                # B->A->B
                cycleB = self.GeneratorA(outputA)
                # A->B->A
                cycleA = self.GeneratorB(outputB)
                

                # Fake output of Discriminator high and low Frequency
                fake_A_h, fake_A_l = self.DiscriminatorA(outputA)
                fake_B_h, fake_B_l = self.DiscriminatorB(outputB)
                
                # Real output of Discriminator high and Low Frequency
                real_A_h, real_A_l = self.DiscriminatorA(imgA)
                real_B_h, real_B_l = self.DiscriminatorB(imgB)

                
                #### Train Generator ####        
                self.optimGA.zero_grad()
                self.optimGB.zero_grad()

                # High, Low Frequency Loss
                h_loss = self.criterion(fake_A_h, self.real_h) + self.criterion(fake_B_h, self.real_h)
                l_loss = self.criterion(fake_A_l, self.real_l) + self.criterion(fake_B_l, self.real_l)

                cycle_loss = (self.cycleloss(imgA, cycleA) + self.cycleloss(imgB, cycleB)) * self.cycleLambda
                
                # Generator Loss
                generator_loss = h_loss * p * 2 + l_loss * (1 - p) * 2 + cycle_loss
                generator_loss.backward(retain_graph=True)

                self.optimGA.step()
                self.optimGB.step()



                #### Train Discriminator ####        
                self.optimDA.zero_grad()
                self.optimDB.zero_grad()

                # A->B
                outputB = self.GeneratorA(imgA)
                # B->A
                outputA = self.GeneratorB(imgB)

                # Imagepool
                outputA, _ = ImagePool(self.ImagepoolA, outputA)
                outputB, _ = ImagePool(self.ImagepoolB, outputB)

                # Fake output of Discriminator high and low Frequency
                fake_A_h, fake_A_l = self.DiscriminatorA(outputA)
                fake_B_h, fake_B_l = self.DiscriminatorB(outputB)

                # High, Low Frequency Loss
                fake_h_loss = self.criterion(fake_A_h, self.fake_h) + self.criterion(fake_B_h, self.fake_h)
                fake_l_loss = self.criterion(fake_A_l, self.fake_l) + self.criterion(fake_B_l, self.fake_l)

                real_h_loss = self.criterion(real_A_h, self.real_h) + self.criterion(real_B_h, self.real_h)
                real_l_loss = self.criterion(real_A_l, self.real_l) + self.criterion(real_B_l, self.real_l)
                

                # Discriminator Loss
                discriminator_loss = (real_h_loss + fake_h_loss) * p * 2 + (real_l_loss + fake_l_loss) * (1 - p) * 2
                discriminator_loss.backward(retain_graph=True)

                self.optimDA.step()
                self.optimDB.step()
                msg = (
                    '[{:03}/{}]'.format(epoch,self.Epoch) + 
                    '[ D : {:.3f}]'.format(discriminator_loss.item()) + 
                    '[ G : {:.3f}]'.format(generator_loss.item())+
                    '[ Cycle : {:.3f}]'.format(cycle_loss.item())
                )

                pbar.set_description_str(msg)

            self.save()