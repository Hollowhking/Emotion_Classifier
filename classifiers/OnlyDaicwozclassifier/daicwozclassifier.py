import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm
import sys
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import lightning as pl
import torch.nn.functional as F
import lightning.pytorch
from lightning.pytorch import Trainer
from sklearn.preprocessing import LabelEncoder
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import Accuracy
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



#================

#quit()
#MAKING THE LIGHTNING MODEL:
class ligNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hid_size, num_classes):
        super(ligNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hid_size)
        self.sigmoid= nn.Sigmoid()
        self.l2 = nn.Linear(hid_size, hid_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hid_size, num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        out = self.sigmoid(self.l1(x))
        out = self.l2(out)
        out = self.l3(out)
        return out

    def training_step(self, batch, batch_idx):
        features, target = batch
        outputs = self(features)
        loss = F.cross_entropy(outputs, target)
        self.accuracy(outputs, target)
        self.log('acc', self.accuracy, prog_bar=True)
        self.log('loss', loss) 
        return loss 

    # def train_dataloader(self):
    #     return DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self.forward(inputs)
        self.accuracy(outputs, target)
        self.log('val_acc', self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
#=============================


#SIZES:
#hyper params:
input_size = 88
hidden_size = 1500
num_classes = 2
num_epochs = 100
batchsize = 128
lr = 0.001 
#=====
#MODIFYING THE DATASET:
df = pd.read_csv("fulldaicwoz.csv")
scaler = StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
# Split data
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
train_size = int(0.7 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size
train_data, val_data, test_data = random_split(CustomDataset(X, y), [train_size, val_size, test_size])

#create Trainer & logger::
logging = CSVLogger('./lightning_logs/', 'lstm')
trainer = pl.Trainer(max_epochs=num_epochs, logger=logging)

#TRAIN THE MODEL:
model = ligNeuralNet(input_size, hidden_size, num_classes)
trainer.fit(model, DataLoader(train_data, batch_size= batchsize, num_workers = 4), DataLoader(val_data, batch_size=batchsize, num_workers = 4))


# trainer.predict(model, return_predictions=False)
#================

#get the data:
metrics = pd.read_csv('./lightning_logs/lstm/version_0/metrics.csv')
val_acc = metrics['val_acc'].dropna().reset_index(drop=True).to_frame()
val_acc.index.name = 'epochs'
val_acc.columns = ['LSTM_acc']
print(val_acc)