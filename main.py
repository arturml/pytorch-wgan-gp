import argparse
import torch
import pandas as pd
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from models import Generator, Discriminator
from wgangp import WGANGP

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--use_cuda',  type=str, default='True')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

if args.dataset == 'mnist':
    train_dataset = MNIST(root='data/mnist', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='data/mnist', train=False, download=True, transform=transform)

if args.dataset == 'fashion':
    train_dataset = FashionMNIST(root='data/fashion', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='data/fashion', train=False, download=True, transform=transform)

full_dataset = ConcatDataset([train_dataset, test_dataset])
data_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

generator = Generator(100)
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

wgan = WGANGP(generator, discriminator, g_optimizer, d_optimizer, [100, 1, 1], args.dataset)
wgan.train(data_loader, args.epochs)

pd.DataFrame(wgan.hist).to_csv(args.dataset + '_hist.csv', index=False)
