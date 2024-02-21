#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to make a **very** basic transformer model that predicts load values. For this initial model, I intend to **only** use previous readings as input. That is, **no context variables (weather, time, etc)** will be used here.

# In[16]:


import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import pandas as pd
import numpy as np
import random


# In[17]:


# -------
EPOCHS = 10
LR = 0.001
SEQ_LENGTH = 12 # Number of historical data points to consider
BATCH_SIZE = 64
D_MODEL = 1 #2  # number of features (demand + temperature)
NHEAD = 1
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
# -------


# In[18]:


def create_sequences(data, seq_length, num_samples):
    sequences = []
    target = []
    if (num_samples > len(data)):
        print("num_samples too large")
        return
    
    for _ in range(num_samples):
        idx = random.randint(0, len(data)-seq_length - 1)
        seq = data[idx:idx+seq_length]
        label = data[idx+seq_length]
        sequences.append(seq)
        target.append(label)

    return np.array(sequences), np.array(target)


# In[19]:


class TransformerPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        output = self.transformer_encoder(src)  # Shape: (SEQ_LENGTH, BATCH_SIZE, D_MODEL)
        last_output = output[-1, :, :]          # Shape: (BATCH_SIZE, D_MODEL)
        prediction = self.fc(last_output)       # Shape: (BATCH_SIZE, 1)
        return prediction.squeeze()             # Shape: (BATCH_SIZE,)


# In[33]:


# Load your data into a DataFrame
path = "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/electricgrid/data/final_tables/banc/banc.csv"
df = pd.read_csv(path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Z-score normalization
mean_demand = df['Demand (MWh)'].mean()
std_demand = df['Demand (MWh)'].std()

df['Normalized Demand'] = (df['Demand (MWh)'] - mean_demand) / std_demand


# In[34]:


# Create sequences
X, y = create_sequences(df['Normalized Demand'].values, SEQ_LENGTH, 5000)  # For example, creating 5000 samples
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


# In[35]:


# Split data (80/20 split)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[36]:


X_train


# In[37]:


# Data loaders
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)


# In[38]:


# Model, Loss, Optimizer
model = TransformerPredictor(D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# In[39]:


# Training loop
model.train()
for epoch in range(EPOCHS):
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item()}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:





# In[9]:


df


# In[ ]:




