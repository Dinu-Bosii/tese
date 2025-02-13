import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from snntorch import utils, spikegen
from snn_model import compute_loss
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import copy

# Temporal Dynamics
#num_steps = 10

# NN Architecture
class CSNNet(nn.Module):
    def __init__(self, input_size,num_steps, beta, spike_grad=None, num_outputs=2):
        super().__init__()
        self.num_steps = num_steps
        self.max_pool_size = 2
        self.conv_kernel = 3 #5, -6
        self.conv_stride = 1 #1
        self.conv_groups = 1 #
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel, stride=self.conv_stride, padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1.5, learn_threshold=True)
        self.conv2 = nn.Conv1d(in_channels=self.conv1.out_channels, out_channels=8, kernel_size=self.conv_kernel, stride=self.conv_stride,groups=self.conv_groups, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True)

        lin_size = self.calculate_lin_size(input_size)

        self.fc_out = nn.Linear(lin_size * self.conv2.out_channels, num_outputs)
        #self.fc_out = nn.Linear(lin_size * self.conv1.out_channels, num_outputs)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True)
    

    def calculate_lin_size(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = F.max_pool1d(self.conv1(x), kernel_size=self.max_pool_size)
        x = F.max_pool1d(self.conv2(x), kernel_size=self.max_pool_size)
        lin_size = x.shape[2]
        return lin_size

    def forward(self, x, input_encoding="rate"):
        if input_encoding == "rate":
            return self.forward_rate(x)
        elif input_encoding == "ttfs":
            return self.forward_ttfs(x)
        else:
            raise ValueError(f"Error in input encoding type.")

    def forward_rate(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem_out = self.lif_out.reset_mem()
        #utils.reset(self)

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for _ in range(self.num_steps): #adicionar prints
            cur1 = F.max_pool1d(self.conv1(x), kernel_size=self.max_pool_size)
            spk, mem1 = self.lif1(cur1, mem1) 

            #print("1st layer out: ", spk.shape) # deve ser mais de 150/200
            #print("1st layer out (flat): ",spk.flatten().shape)

            cur2 = F.max_pool1d(self.conv2(spk), kernel_size=self.max_pool_size)
            spk, mem2 = self.lif2(cur2, mem2)
            #print("2nd layer out: ", spk.shape) # deve ser mais de 150/200
            
            spk = spk.view(spk.size()[0], -1)
            #print("2nd layer out (flat): ", spk.shape)
            #print(self.lif_out)
            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            #print(spk_out.size())
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    
    def forward_ttfs(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem_out = self.lif_out.reset_mem()
        #utils.reset(self)

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        in_spikes = spikegen.latency(x, num_steps=self.num_steps, linear=True)
        #print(in_spikes.size())

        for x_in in in_spikes: #adicionar prints
            cur1 = F.max_pool1d(self.conv1(x_in), kernel_size=self.max_pool_size)
            spk, mem1 = self.lif1(cur1, mem1) 

            #print("1st layer out: ", spk.shape) # deve ser mais de 150/200
            #print("1st layer out (flat): ",spk.flatten().shape)

            cur2 = F.max_pool1d(self.conv2(spk), kernel_size=self.max_pool_size)
            spk, mem2 = self.lif2(cur2, mem2)
            #print("2nd layer out: ", spk.shape) # deve ser mais de 150/200
            
            spk = spk.view(spk.size()[0], -1)
            #print("2nd layer out (flat): ", spk.shape)
            #print(self.lif_out)
            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            #print(spk_out.size())
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)


        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    

def train_csnn(net, optimizer,  train_loader, val_loader, train_config, net_config):
    device, num_epochs, num_steps = train_config['device'],  train_config['num_epochs'], train_config['num_steps']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    batch_size = train_config['batch_size']
    val_fn = train_config['val_net']
    loss_hist = []
    val_acc_hist = []
    val_auc_hist = []
    best_auc_roc = 0
    best_net_list = []
    auc_roc = 0
    loss_val = 0
    for epoch in range(num_epochs):
        net.train()
        #if (epoch + 1) % 10 == 0: print(f"Epoch:{epoch + 1}|auc:{auc_roc}|loss:{loss_val.item()}")

        # Minibatch training loop
        for data, targets in train_loader:
            #print(data.size(), data.unsqueeze(1).size())

            data = data.to(device, non_blocking=True).unsqueeze(1)

            targets = targets.to(device, non_blocking=True)
            #print(targets.size(), targets.unsqueeze(1).size())
            # forward pass
            spk_rec, mem_rec = net(data)
            #print(spk_rec, mem_rec)

            # Compute loss
            loss_val = compute_loss(loss_type=loss_type, loss_fn=loss_fn, spk_rec=spk_rec, mem_rec=mem_rec,num_steps=num_steps, targets=targets, dtype=dtype, device=device) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            #loss_hist.append(loss_val.item())
       # _, auc_roc = val_fn(net, device, val_loader, train_config)
        #if auc_roc > best_auc_roc:
        #    best_auc_roc = auc_roc
        #print(f"Epoch:{epoch + 1} - auc:{auc_roc} - loss:{loss_val}")           
        best_net_list.append(copy.deepcopy(net.state_dict()))

            #val_acc_hist.extend(accuracy)
        #val_auc_hist.extend([auc_roc])


    return net, loss_hist, val_acc_hist, val_auc_hist, best_net_list


def val_csnn(net, device, val_loader, train_config):
    eval_batch = iter(val_loader)
    accuracy = 0
    auc_roc = 0
    all_preds = []
    all_targets = []
    batch_size = train_config['batch_size']
    prediction_fn = train_config['prediction_fn']
    net.eval()
    for data, targets in eval_batch:
        data = data.to(device, non_blocking=True).unsqueeze(1)
        data_size = data.shape[0]
        if data.shape[0] < batch_size:
            last_sample = data[-1].unsqueeze(0)
            num_repeat = batch_size - data.shape[0]
            repeated_samples = last_sample.repeat(num_repeat, *[1] * (data.dim() - 1))

            data = torch.cat([data, repeated_samples], dim=0)
        targets = targets.to(device, non_blocking=True)

        spk_rec, mem_rec = net(data)

        spk_rec = spk_rec[:, :data_size]
        mem_rec = mem_rec[:, :data_size]

        predicted = prediction_fn(spk_rec)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())


    ## accuracy and roc-auc
    accuracy = accuracy_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)
    
    return accuracy, auc_roc


def test_csnn(net,  device, test_loader, train_config):
    all_preds = []
    all_targets = []
    batch_size = train_config['batch_size']
    prediction_fn = train_config['prediction_fn']
    # Testing Set Loss
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device, non_blocking=True).unsqueeze(1)
            data_size = data.shape[0]
            if data.shape[0] < batch_size:
                last_sample = data[-1].unsqueeze(0)
                num_repeat = batch_size - data.shape[0]
                repeated_samples = last_sample.repeat(num_repeat, *[1] * (data.dim() - 1))
                data = torch.cat([data, repeated_samples], dim=0)

            targets = targets.to(device, non_blocking=True)
            # forward pass
            #test_spk, _ = net(data.view(data.size(0), -1))
            test_spk, _ = net(data)
            test_spk = test_spk[:, :data_size]

            # calculate total accuracy
            # max(1) -> gives the index (either 0 or 1) for either output neuron 
            # based on the times they spiked in the 10 time step interval

            predicted = prediction_fn(test_spk)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets
    

def prediction_spk_rate_pop(spk_rec):
    spk_sum = spk_rec.sum(dim=0)

    population_c0 = spk_sum[:,:spk_rec.shape[2]//2].mean(dim=1)
    population_c1 = spk_sum[:,spk_rec.shape[2]//2:].mean(dim=1)

    predicted = torch.stack([population_c0, population_c1], dim=1)
    _, predicted = predicted.max(1)
    return predicted


def prediction_spk_rate(spk_rec):
    _, predicted = spk_rec.sum(dim=0).max(1)
    return predicted


def prediction_spk_ttfs(spk_rec):
    return spk_rec.argmax(dim=0).min(1).indices


def get_prediction_fn(encoding='rate', pop_coding=False):
    if encoding == 'rate':
        if pop_coding:
            return prediction_spk_rate_pop
        else:
            return prediction_spk_rate
    elif encoding == 'ttfs':
        #no support for pop_coding yet
        return prediction_spk_ttfs
    else:
        raise ValueError('Invalid spike prediction fn.')

    

"""        #rate
        #_, predicted = spk_rec.sum(dim=0).max(1)
        #print("spk_rec", spk_rec.size())
        #population coding - wip
        predicted = spk_rec.sum(dim=0)
        #print("pred_sum:", predicted.size())
        predicted_c0 = predicted[:,:5]
        #print("predicted c0:", predicted_c0.size())
        predicted_c0 = predicted_c0.mean(dim=1)  # 5 should be dynamic (spk_rec.shape[2]//2)
        #print("predicted c0 mean:", predicted_c0.size())

        predicted_c1 = predicted[:,5:].mean(dim=1)
        predicted = torch.stack([predicted_c0, predicted_c1], dim=1)
        #.argmax(dim=0)
        _, predicted = predicted.max(1)
        #print("predicted:", predicted.size())
        #temporal
        #predicted = spk_rec.argmax(dim=0).min(1).indices  # Get first spike time for each neuron
"""