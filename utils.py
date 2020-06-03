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


def ImagePool(pool, image,device = torch.device('cuda'), max_size = 50):

    gen_image = image.cpu().detach().numpy()
    if len(pool) < max_size:
        pool.append(gen_image)
        return image, True
    else:
        p = random()
        if p > 0.5:
            random_id = randint(0, len(pool)-1)
            temp = pool[random_id]
            pool[random_id] = gen_image
            return torch.Tensor(temp).to(device), True
        else:
            return torch.Tensor(gen_image).to(device), False
        
        
class dataset(Dataset):
    def __init__(self, dir, image_size):
        self.transform = transforms.Compose([
                                             transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor()
        ])
        self.imglist = os.listdir(dir)
        self.dir = dir


    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = Image.open(self.dir+'/'+ self.imglist[idx])
        img = self.transform(img)

        return img
    
    
    
def saveimage(image, epoch, path, figsize = (25,25), x=5, label = True, name = True, t = 1):
    path = path + '{}/{}_{:03}.jpg'.format(t,t, epoch)
    fig, ax = plt.subplots(x, x, figsize=figsize)
    
    image = np.transpose(image.cpu().detach().numpy(), (0, 2, 3, 1))
    
    for i, img in enumerate(image):
        ax[int(i/x), i%x].imshow(img)
        ax[int(i/x), i%x].get_xaxis().set_visible(False)
        ax[int(i/x), i%x].get_yaxis().set_visible(False)

    if name:
        del(name)
        name = '{} Try Epoch {}'.format(t, epoch)
    if label:
        fig.text(0.5, 0.04, name, ha='center', fontsize = 15)
    plt.savefig(path)
    del(fig)
    del(ax)
    plt.close()