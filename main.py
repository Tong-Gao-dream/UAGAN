import torch.optim
import os

from train import train
from torch.utils.data import DataLoader
from util.read_data import SegmentationDataset
from model.Generator import Generator
from model.Discriminator import Discriminator
from config import input_args

os.makedirs('./save_model/save_G', exist_ok=True)
os.makedirs('./save_model/save_D', exist_ok=True)

args = input_args()
print('args', args)

dataset = SegmentationDataset(args.image_dir, args.mask_dir)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

generator = Generator().cuda()
discriminator = Discriminator().cuda()

optim_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

loss_adv = torch.nn.BCELoss().cuda()
loss_rec = torch.nn.MSELoss().cuda()

train(args, dataloader, generator, discriminator,optim_G, optim_D, loss_adv, loss_rec)