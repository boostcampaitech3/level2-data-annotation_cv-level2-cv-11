import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tqdm import tqdm
from argparse import ArgumentParser

from utils import *
from dataset import CityScapeDataset
from generator import Generator
from discriminator import Discriminator

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_dir', default='cityscapes/train')
    parser.add_argument('--val_dir', default='cityscapes/val')
    parser.add_argument('--l1_lambda', type=int, default=100)
    args = parser.parse_args()

    return args

def train_fn(disc, gen, opt_disc, opt_gen, l1, bce, loader, args):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(args.device), y.to(args.device)

        #########################
        ## Train Discriminator ##
        #########################
        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach()) # computational graph 유지

        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake_loss = bce(D_fake, torch.zeros_like(D_real))

        D_loss = (D_real_loss + D_fake_loss) / 2 # 

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

def main(args):
    discriminator_model = Discriminator(in_channels=3).to(args.device)
    generator_model = Generator(in_channels=3).to(args.device)

    optimizer_discriminator = optim.Adam(discriminator_model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_generator = optim.Adam(generator_model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    train_dataset = CityScapeDataset(root_dir=args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_dataset = CityScapeDataset(root_dir=args.val_dir)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for epoch in range(args.num_epochs):
        train_fn(discriminator_model, generator_model, optimizer_discriminator, optimizer_generator, criterion_bce, criterion_l1, train_loader, args)

        if epoch % 5== 0:


if __name__ == "__main__":
    args = parse_args()
    main(args)

