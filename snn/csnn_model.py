import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import ce_rate_loss, ce_temporal_loss, ce_count_loss
import torch.nn.functional as F
from snntorch import utils
from snn_model import compute_loss
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

# Neurons Count
num_outputs = 2

# Temporal Dynamics
#num_steps = 10
beta = 0.95

# NN Architecture
class CSNNet(nn.Module):
    def __init__(self, input_size,num_steps, spike_grad=None):
        super().__init__()
        self.num_steps = num_steps
        #trocar out channels - diminuir
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv1d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # The formula below for calculate output size of conv doesn't always give the correct otput
        # n_out = ((n_in  + 2 * padding - kernel_size) // stride) + 1
        #lin_size = (input_size + 2 * 1 - 5) // 1 + 1
        #lin_size = lin_size//2

        #lin_size = (lin_size + 2 * 1 - 3) // 1 + 1
        #lin_size = lin_size//2

        lin_size = self.calculate_lin_size(input_size)

        #evitar embeddings pequenos a entrar na linear
        self.fc_out = nn.Linear(lin_size * self.conv2.out_channels, num_outputs)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
    

    def calculate_lin_size(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = F.max_pool1d(self.conv1(x), kernel_size=2)
        x = F.max_pool1d(self.conv2(x), kernel_size=2)
        lin_size = x.shape[2]
        return lin_size


    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem_out = self.lif_out.reset_mem()
        #utils.reset(self)

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for _ in range(self.num_steps): #adicionar prints
            cur1 = F.max_pool1d(self.conv1(x), kernel_size=2)
            spk, mem1 = self.lif1(cur1, mem1) 

            #print("1st layer out: ", spk.shape) # deve ser mais de 150/200
            #print("1st layer out (flat): ",spk.flatten().shape)

            cur2 = F.max_pool1d(self.conv2(spk), kernel_size=2)
            spk, mem2 = self.lif2(cur2, mem2)
            #print("2nd layer out: ", spk.shape) # deve ser mais de 150/200
            
            spk = spk.view(spk.size()[0], -1)
            #print("2nd layer out (flat): ", spk.shape)
            #print(self.lif_out)
            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    

def train_csnn(net, optimizer, train_loader, val_loader, train_config):
    device, num_epochs, num_steps = train_config['device'],  train_config['num_epochs'], train_config['num_steps']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    loss_hist = []
    val_acc_hist = []
    val_auc_hist = []

    for epoch in range(num_epochs):
        net.train()
        print(f"Epoch:{epoch + 1}")

        # Minibatch training loop
        for data, targets in train_loader:
            data = data.to(device).unsqueeze(1)
            targets = targets.to(device)
            #print(data.size(), data.view(batch_size, -1).size())
            if data.shape[0] < 32:
                continue
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)
            #print(spk_rec, mem_rec)

            # Compute loss
            loss_val = compute_loss(loss_type=loss_type, loss_fn=loss_fn, spk_rec=spk_rec, mem_rec=mem_rec,num_steps=num_steps, targets=targets, dtype=dtype) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
        
        #_, val_acc, val_auc = val_csnn(net, device, val_loader, loss_type,loss_fn, num_steps, dtype)
        # 
        #val_acc_hist.append(val_acc)
        #val_auc_hist.append(val_auc)

    return net, loss_hist, val_acc_hist, val_auc_hist


def val_csnn(net, device, val_loader, loss_type, loss_fn, num_steps, dtype):
    all_preds = []
    all_targets = []
    mean_loss = 0
    net.eval()
    # Minibatch training loop
    i = 0
    for data, targets in val_loader:
        i+=1
        data = data.to(device).unsqueeze(1)
        if data.shape[0] < 32:
            continue
        targets = targets.to(device)

        spk_rec, mem_rec = net(data)

        loss_val = compute_loss(loss_type=loss_type, loss_fn=loss_fn, spk_rec=spk_rec, mem_rec=mem_rec,num_steps=num_steps, targets=targets, dtype=dtype) 
        _, predicted = spk_rec.sum(dim=0).max(1)
            
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # Store loss history
        mean_loss += loss_val
    
    return mean_loss/i, accuracy_score(all_targets, all_preds), roc_auc_score(all_targets, all_preds)


def test_csnn(net,  device, test_loader):
    all_preds = []
    all_targets = []

    # Testing Set Loss
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device).unsqueeze(1)
            targets = targets.to(device)
            if data.shape[0] < 32:
                continue
            # forward pass
            #test_spk, _ = net(data.view(data.size(0), -1))
            test_spk, _ = net(data)

            # calculate total accuracy
            # max(1) -> gives the index (either 0 or 1) for either output neuron 
            # based on the times they spiked in the 10 time step interval
            
            _, predicted = test_spk.sum(dim=0).max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets
    
