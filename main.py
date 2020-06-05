import torch
import torch.nn as nn
import argparse
import cyclegan



parser = argparse.ArgumentParser()


parser.add_argument('--device'          ,default = torch.device('cuda')      ,help = 'If you want cpu, cuda -> cpu')

parser.add_argument('--Epoch'           ,default = 100                       ,help = 'Total Epoch')
parser.add_argument('--batch_size'      ,default = 4                         ,help = 'Batch Size')
parser.add_argument('--lr'              ,default = 0.0002                    ,help = 'Learning Rate')
parser.add_argument('--alpha'           ,default = 0.125                     ,help = 'Separating Coefficient')
parser.add_argument('--num'             ,default = 0                         ,help = 'Try Number')


parser.add_argument('--criterion'       ,default = nn.MSELoss()              ,help = 'If you want BCE Loss, Change nn.BCELoss()')
parser.add_argument('--cycleloss'       ,default = nn.L1Loss()               ,help = 'Cycle Consistency Loss')

parser.add_argument('--cyclelambda'     ,default = 10                        ,help = 'Cycle Loss Coefficient')
parser.add_argument('--image_size'      ,default = 256                       ,help = 'Image Size')
parser.add_argument('--num_workers'     ,default = 2                         ,help = 'Dataloader num_workers')
parser.add_argument('--betas'           ,default = (0.5, 0.999)              ,help = 'Adam Optimizer Betas')
parser.add_argument('--test_batch_size' ,default = 25                        ,help = 'Want Test Data Number')

parser.add_argument('--shuffle'         ,default = True                      ,help = 'Dataloader shuffle')
parser.add_argument('--random'          ,default = False                     ,help = 'Want High/Low Frequency Division')

parser.add_argument('--output_path'     ,default = 'output/'                 ,help = 'Path for Output Image')
parser.add_argument('--dirA'            ,default = 'data/monet2photo/trainA' ,help = 'Domain A Train Data Path')
parser.add_argument('--dirB'            ,default = 'data/monet2photo/trainB' ,help = 'Domain B Train Data Path')
parser.add_argument('--dirA_test'       ,default = 'data/monet2photo/testA'  ,help = 'Domain A Test Data Path')
parser.add_argument('--dirB_test'       ,default = 'data/monet2photo/testB'  ,help = 'Domain B Test Data Path')


#args = parser.parse_args()
## If you use .ipynb, I recommend below code
args, _ = parser.parse_known_args()

CycleGAN = cyclegan.CycleGAN(args)


def train():
    CycleGAN.train()

    
if __name__=='__main__':
    train()
    
    