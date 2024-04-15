import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import sys
from sklearn.preprocessing import OneHotEncoder #turn the y from "anger" into a binary list 

df = pd.read_csv("audiodatacomplete.csv")


df = df.sample(frac = 1, random_state= 69)#shuffle rows

#Y_whole = df['Emotion'].tolist()
X_whole = df.iloc[:,1:88]
Y_whole = df.iloc[:,0]

Y_whole = Y_whole.to_numpy()
Y_whole = Y_whole.reshape(-1,1)

#encode our Y
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
Y_whole = encoder.fit_transform(Y_whole)
#print(Y_whole[0:5])
#turns it to:
# [[0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]]
# Split data into train and test sets
test_data_break = 5000
X_train = torch.tensor(X_whole[:test_data_break].values, dtype=torch.float32)
Y_train = torch.tensor(Y_whole[:test_data_break], dtype=torch.float32)
X_test = torch.tensor(X_whole[test_data_break:].values, dtype=torch.float32)
Y_test = torch.tensor(Y_whole[test_data_break:], dtype=torch.float32)


#sys.exit()
#should use a softmax output function
class multiclassifier(nn.Module):#88 inputs -> 176 hidden neurons -> 5 outputs
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(87,176)
        self.act = nn.Sigmoid()#non linear activations
        self.output = nn.Linear(176,5)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

model = multiclassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
batch_size = 400
batches_per_epoch = len(X_train)

train_lost_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

#training loop:
for epoch in range(num_epochs):
    epoch_loss = []
    epoch_acc = []
    model.train()
    for i in range(0, len(X_train), batch_size):

        optimizer.zero_grad()

        # grab a batch
        X_batch = X_train[i:i+batch_size]
        Y_batch = Y_train[i:i+batch_size]

        #forward pass:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred,Y_batch)

        #backward pass
        loss.backward()

        #update weights
        optimizer.step()

        #compute and store
        acc = (torch.argmax(y_pred, 1) == torch.argmax(Y_batch, 1)).float().mean().item()
        epoch_loss.append(loss.item())
        epoch_acc.append(acc)
        
    #set model in eval mode run through test set:
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        #print(y_pred)
        ce = loss_fn(y_pred, torch.max(Y_test, 1)[1])
        acc = (torch.argmax(y_pred, 1) == torch.argmax(Y_test, 1)).float().mean().item()

        train_lost_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce.item())
        test_acc_hist.append(acc)

        print(f"Epoch {epoch} validation: Cross-entropy={ce.item():.2f}, Accuracy={acc*100:.1f}%")

#test = model()
# Restore best model