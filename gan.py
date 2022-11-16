import os
import random
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class GAN:
    def __init__(self, EPOCHS=500, BATCH_SIZE=100, GPU_ID=1, SEED=2022):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.GPU_ID = GPU_ID
        self.SEED = SEED
        start_time = datetime.datetime.now().strftime("%m_%d/%H_%M_%S")
        self.writer = SummaryWriter(f'tensorboard/{start_time}')
        
        print(f"Start training at {start_time}")
        
        # Fix Seed
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        np.random.seed(self.SEED)
        random.seed(self.SEED)

        # GPU Settings
        self.DEVICE = torch.device(self.GPU_ID if torch.cuda.is_available() else "cpu")
        print("Available Devices:", torch.cuda.device_count())
        print("Using Device:", self.DEVICE)

        
    def dataloader(self, mode="train"):
        if mode == "train":
            self.trainset = datasets.FashionMNIST(
                './.data',
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )

            self.train_loader = torch.utils.data.DataLoader(
                dataset = self.trainset,
                batch_size = self.BATCH_SIZE,
                shuffle = True
            )
            
    def init_networks(self):
        G = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
        D = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.D = D.to(self.DEVICE)
        self.G = G.to(self.DEVICE)

    def train(self):
        self.dataloader(mode="train")
        self.init_networks()
        
        self.criterion = nn.BCELoss()
        self.d_opt = optim.Adam(self.D.parameters(), lr=1e-4)
        self.g_opt = optim.Adam(self.G.parameters(), lr=1e-4)

        for epoch in range(1, self.EPOCHS+1):
            train_iter = len(self.train_loader)
            
            tqdm_batch = tqdm(self.train_loader, total=train_iter, bar_format="{l_bar}{bar:20}{r_bar}", 
                            desc=f"Train-Epoch {epoch}", disable=False)
            
            for i, (images, _) in enumerate(tqdm_batch):
                images = images.reshape(self.BATCH_SIZE, -1).to(self.DEVICE)
                
                real_labels = torch.ones(self.BATCH_SIZE, 1).to(self.DEVICE)
                fake_labels = torch.zeros(self.BATCH_SIZE, 1).to(self.DEVICE)
                
                # Training Discriminator
                outputs = self.D(images)
                d_loss_real = self.criterion(outputs, real_labels)
                real_score = outputs
                
                z = torch.randn(self.BATCH_SIZE, 64).to(self.DEVICE)
                fake_images = self.G(z)
                
                outputs = self.D(fake_images)
                d_loss_fake = self.criterion(outputs, fake_labels)
                fake_score = outputs
                
                d_loss = d_loss_fake + d_loss_real
                
                self.d_opt.zero_grad()
                self.g_opt.zero_grad()
                
                d_loss.backward()
                self.d_opt.step()
                
                # Training Generator
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.criterion(outputs, real_labels)
                
                self.d_opt.zero_grad()
                self.g_opt.zero_grad()
                
                g_loss.backward()
                self.g_opt.step()
            tqdm_batch.close()
            
            self.writer.add_scalar("train/Loss_D", d_loss.item() ,epoch)
            self.writer.add_scalar("train/Loss_G", g_loss.item() ,epoch)
            self.writer.add_scalar("train/D(x)", real_score.mean() ,epoch)
            self.writer.add_scalar("train/D(G(z))", fake_score.mean() ,epoch)
            self.writer.add_image("train/Image", fake_images[:2].reshape(2, 28, 28), epoch)
            # self.writer.flush()
                
            print("Epoch [{}/{}], d_loss: {:.4f}, g_loss:{:.4f}, D(x):{:.2f}, D(G(z)):{:.2f}"
                .format(epoch, self.EPOCHS, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))
        
        self.writer.close()