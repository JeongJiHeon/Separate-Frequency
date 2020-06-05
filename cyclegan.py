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
from tqdm import tqdm
from random import *

from torchvision import models
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from network import *
from utils import *




class CycleGAN():
    def __init__(self, args):        
        
        """
        ----------------------------------------------------------------------------------------
        
        Set Function:
            save  : Save Model
            load  : Load Model
            train : train Model
            

        ----------------------------------------------------------------------------------------
        """        
        
        self.Epoch           = args.Epoch
        self.lr              = args.lr
        self.cycleloss       = args.cycleloss
        self.cyclelambda     = args.cyclelambda
        self.batch_size      = args.batch_size
        self.criterion       = args.criterion
        self.random          = args.random
        self.betas           = args.betas
        self.path            = args.output_path
        self.device          = args.device
        self.num             = args.num
        self.test_batch_size = args.test_batch_size
        
        
        
        
        self.dataloaderA = torch.utils.data.DataLoader(dataset(args.dirA, image_size = args.image_size), shuffle = args.shuffle,
                                                       batch_size = self.batch_size, num_workers = args.num_workers)
        self.dataloaderB = torch.utils.data.DataLoader(dataset(args.dirB, image_size = args.image_size), shuffle = args.shuffle, 
                                                       batch_size = self.batch_size, num_workers = args.num_workers)


#        self.fixdataA = next(iter(torch.utils.data.DataLoader(dataset(args.dirA_test, image_size = args.image_size), 
#                                                              batch_size = self.test_batch_size))).to(device = self.device)
        self.fixdataB = next(iter(torch.utils.data.DataLoader(dataset(args.dirB_test, image_size = args.image_size), 
                                                              batch_size = self.test_batch_size))).to(device = self.device)

        # Generator
        self.GeneratorA = Generator(n = 9 if args.image_size==256 else 6).to(device = self.device) # make B (input A)
        self.GeneratorB = Generator(n = 9 if args.image_size==256 else 6).to(device = self.device) # make A (input B)

        # Discriminator
        self.DiscriminatorA = Discriminator(a = args.alpha).to(device = self.device) # input A
        self.DiscriminatorB = Discriminator(a = args.alpha).to(device = self.device) # input B

        # Optimizer
        self.optimGA = torch.optim.Adam(self.GeneratorA.parameters(), lr = self.lr, betas = args.betas)
        self.optimGB = torch.optim.Adam(self.GeneratorB.parameters(), lr = self.lr, betas = args.betas)
        self.optimDA = torch.optim.Adam(self.DiscriminatorA.parameters(), lr = self.lr, betas = args.betas)
        self.optimDB = torch.optim.Adam(self.DiscriminatorB.parameters(), lr = self.lr, betas = args.betas)



        self.real_h = torch.ones(self.batch_size, args.image_size//16, args.image_size//16).to(device = self.device)
        self.real_l = torch.ones(self.batch_size, args.image_size//32, args.image_size//32).to(device = self.device)

        self.fake_h = torch.zeros(self.batch_size, args.image_size//16, args.image_size//16).to(device = self.device)
        self.fake_l = torch.zeros(self.batch_size, args.image_size//32, args.image_size//32).to(device = self.device)



        self.ImagepoolA = []
        self.ImagepoolB = []
        
        if not os.path.exists(self.path + '{}'.format(self.num)):
            os.mkdir(self.path + '{}'.format(self.num))


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


    def _set_lr(self, epoch):

        # learning rate decreases linearly to zero if epoch > EPOCH//2

        self.optimGA.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)
        self.optimGB.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)
        self.optimDA.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)
        self.optimDB.param_groups[0]['lr'] = self.lr - self.lr * max(epoch - (self.Epoch//2),0)/(self.Epoch//2)



    def train(self, step = 0):
        saveimage(self.fixdataB, self.Epoch, num = self.num, path = self.path)
        for epoch in range(self.Epoch):
            if step > epoch:
                continue

            saveimage(self.GeneratorB(self.fixdataB), epoch, num = self.num, path = self.path)
            self._set_lr(epoch)




            pbar = tqdm(enumerate(zip(self.dataloaderA, self.dataloaderB)), total = len(self.dataloaderA))

            for idx,(imgA, imgB) in pbar:

                p = random() if self.random else 0.5


                imgA = imgA.to(device = self.device)
                imgB = imgB.to(device = self.device)

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

                cycle_loss = (self.cycleloss(imgA, cycleA) + self.cycleloss(imgB, cycleB)) * self.cyclelambda
                
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