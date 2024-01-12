import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import lightning as pl
import torch.nn.functional as F
import lightning.pytorch
from lightning.pytorch import Trainer

#hyper params:
input_size = 88
hidden_size = 500
num_classes = 5
num_epochs = 3
batch_size = 32
lr = 0.001

class ligNeuralNet(pl.LightningModule):
    def __init__(self,input_size,hid_size,num_classes):
        super(ligNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hid_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hid_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out
    
    def training_step(self, batch, batch_idx):
        features, target = batch

        # Forward pass
        outputs = self(features)
        loss = F.cross_entropy(outputs, target)
        
        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}

    # define what happens for testing here

    def train_dataloader(self):
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=False
        )
        return train_loader
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
if __name__ == '__main__':
    trainer = Trainer(fast_dev_run = True)
    model = ligNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)