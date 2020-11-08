from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


from model.model1 import Generator, Discriminator
import utils
from  data.animeFace import AnimeFaceDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/home/anupam/projects/dataset/anime-faces/', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
parser.add_argument('--imagesize', type=int, default=64, help='height/width of the input image')
parser.add_argument('--nz', type=int, default=100, help='size of latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=64, help='number of trainnig epochs to run')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for training')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outputdir', default='.', help='path to dataset')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adam optimizers')

args = parser.parse_args()

dataset = AnimeFaceDataset(root_dir = args.dataroot,
                         transform = transforms.Compose([
                             transforms.Resize(args.imagesize),
                             transforms.CenterCrop(args.imagesize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batchsize,
                                         shuffle=True, num_workers = args.workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0 ) else "cpu")

#Plot some training training images
'''
real_batch = next(iter(dataloader))
print('anupam')
print(real_batch.size())
'''
##############




netG = Generator(args.ngpu).to(device)
if(device.type == 'cuda' and ngpu>1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(utils.weight_init)


netD = Discriminator(args.ngpu).to(device)
if(device.type == "cuda" and ngpu>1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(utils.weight_init)



#loss
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

#optimizers
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


#training loop

fixed_noise = torch.randn(64, args.nz, 1, 1, device = device)

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

for epoch in range(args.num_epoch):
    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        #print(netD)
        real_cpu = data.to(device)
        #print(real_cpu.size())
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype = torch.float, device = device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        output = netD(real_cpu).view(-1)
        D_x = output.mean().item()


        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()


        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        #print('[%s/%s] [%s/%s]'
        #      % ("epoch", "num_epoch", "batch", "length"))

        if i % 10 == 0:
            print('[%d/%d] [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.num_epoch, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == args.num_epoch-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
     
