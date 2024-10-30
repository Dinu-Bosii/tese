#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from rdkit import Chem
from snn_model import Net, device, batch_size, num_steps
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from snntorch import spikegen, surrogate
import matplotlib.pyplot as plt
from utils import load_dataset_df, fp_generator


# In[2]:


files = ['tox21.csv','sider.csv', 'BBBP.csv']
filename = files[2]
df, targets = load_dataset_df(filename=filename)
print(targets)

target_name = targets[0]
df = df[[target_name, 'smiles']].dropna()


# In[3]:


fp_types = [['morgan', 1024], ['maccs', 167], ['layered', 1024]]
fp_type, num_bits = fp_types[1]

num_rows = len(df)
fp_array = np.zeros((num_rows, num_bits))
target_array = np.zeros((num_rows, 1))
i = 0

# Smile to Morgan Fingerprints of size {num_bits}
fp_gen = fp_generator(fp_type)
for idx, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['smiles'])
    #TODO: sanitize molecules to remove the warnings (?)
    
    if mol is not None:
        fingerprint = fp_gen(mol)

        fp_array[i] = np.array(fingerprint)
        target_array[i] = row[target_name]
        i += 1
target_array = target_array.ravel()


# In[ ]:


# Split the Data- verificar para 30x
dtype = torch.float32
fp_tensor = torch.tensor(fp_array, dtype=dtype)
target_tensor = torch.tensor(target_array, dtype=dtype).long()

dataset = TensorDataset(fp_tensor, target_tensor)
train, test = random_split(dataset, [0.7, 0.3])

_, test_label = test[:]
print("positive labels in the test set:", int(test_label.sum()))

# Load the Data
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

print(len(train), len(test))


# In[18]:


num_epochs = 30
loss_hist = []
test_loss_hist = []
counter = 0

# Load the network
spike_grad=surrogate.fast_sigmoid()
spike_grad=None
net = Net(num_inputs=num_bits, spike_grad=spike_grad).to(device)
loss = nn.CrossEntropyLoss()
#binary cross entropy - ver!
#racio do pos:neg
#BCEWithLogitsLoss: pos_weight parameter

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0)


# In[19]:


#  Training Loop
for epochs in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)
        #print(data.size(), data.view(batch_size, -1).size())
        # forward pass
        net.train()
        spk_rec, mem_rec = net(data)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data)
        #print(spk_rec, mem_rec)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())


# In[ ]:


fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")


# In[ ]:


from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
all_preds = []
all_targets = []

# Testing Set Loss
with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)
    # forward pass
    #test_spk, _ = net(data.view(data.size(0), -1))
    test_spk, _ = net(data)

    # calculate total accuracy
    # max(1) -> gives the index (either 0 or 1) for either output neuron 
    # based on the times they spiked in the 10 time step interval
    
    _, predicted = test_spk.sum(dim=0).max(1)
    
    all_preds.extend(predicted.cpu().numpy())
    all_targets.extend(targets.cpu().numpy())


accuracy = accuracy_score(all_targets, all_preds)
auc_roc = roc_auc_score(all_targets, all_preds)
tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
specificity = tn/(tn + fp)
sensitivity = tp/(tp + fn)


# Print Results
print(f"Accuracy:  {accuracy * 100:.2f}")
print(f"AUC ROC: {auc_roc:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")



# ROC AUC - Done
# Weights
# Hyperparameters
# BBBP / Cider
# Fingerprints (LayeredFP)
# Melhor input para snns?
