import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from snntorch import spikegen
from snn_model import compute_loss
from sklearn.metrics import roc_auc_score, accuracy_score
import copy

# Temporal Dynamics
#num_steps = 10
out_channels = [8, 8]
in_channels = [1, out_channels[0]]
groups = [1, 1]
thresholds = torch.tensor([1.5, 1.0, 1.0], dtype=torch.float32)
learn_th = [True for _ in range(3)]
#learn_th = [False for _ in range(3)]
class CSNNet(nn.Module):
    def __init__(self, net_config):

        super().__init__()
        self.num_steps = net_config["num_steps"]
        self.pool_size = net_config["pool_size"]
        self.conv_kernel = net_config["conv_kernel"]
        self.conv_stride = net_config["conv_stride"]
        self.conv_groups = net_config["conv_groups"]
        self.num_conv = net_config['num_conv']
        self.input_size = net_config["input_size"]
        self.encoding = net_config['encoding']
        self.flatten = nn.Flatten()

        if len(net_config["input_size"]) == 1:
            self.max_pool = F.max_pool1d
        else:
            self.max_pool = F.max_pool2d
        # even -> Conv | odd -> lif
        self.layers = nn.ModuleList()

        for i in range(self.num_conv):
            if len(net_config["input_size"]) == 1:
                conv_layer = nn.Conv1d(in_channels=in_channels[i], 
                                   out_channels=out_channels[i], 
                                   kernel_size=self.conv_kernel, 
                                   stride=self.conv_stride[i], 
                                   padding=1,
                                   bias=net_config['bias'])
            else:
                conv_layer = nn.Conv2d(in_channels=in_channels[i], 
                                   out_channels=out_channels[i], 
                                   kernel_size=self.conv_kernel[i], 
                                   stride=self.conv_stride, 
                                   padding=1,
                                   bias=net_config['bias'])
            self.layers.append(conv_layer)
            torch.nn.init.xavier_uniform_(conv_layer.weight)

            lif_layer = snn.Leaky(beta=net_config["beta"], spike_grad=net_config["spike_grad"], threshold=thresholds[i], learn_threshold=learn_th[i])
            self.layers.append(lif_layer)

        lin_size = self.calculate_lin_size(self.input_size)

        self.fc_out = nn.Linear(lin_size, net_config['out_num'])
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        self.layers.append(self.fc_out)

        self.lif_out = snn.Leaky(beta=net_config["beta"], spike_grad=net_config['spike_grad'], learn_threshold=learn_th[-1])
        self.layers.append(self.lif_out)

        if self.encoding == "rate":
            self.forward = self.forward_rate
        elif self.encoding == "ttfs":
            self.forward = self.forward_ttfs
        else:
            raise ValueError("Error in encoding type.") 

    def calculate_lin_size(self, input_size):
        #print(type(input_size))
        x = torch.zeros(1, 1, *input_size)

        for i in range(0, len(self.layers), 2):
            conv = self.layers[i]
            #x = F.max_pool1d(conv(x), kernel_size=self.pool_size)
            x = self.max_pool(conv(x), kernel_size=self.pool_size)
        x = self.flatten(x)
        return x.shape[1]
    

    def forward_rate(self, x):
        # Initialize hidden states at t=0
        in_spikes = spikegen.rate(x, num_steps=self.num_steps)

        membranes = []
        for layer in self.layers:
            mem = layer.reset_mem() if isinstance(layer, snn.Leaky) else None
            membranes.append(mem)

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for x_in in in_spikes:
            spk = x_in

            for i in range(0, self.num_conv * 2 - 1, 2):
                conv = self.layers[i]
                lif = self.layers[i+1]

                #cur = F.max_pool1d(conv(spk), kernel_size=self.pool_size)
                cur = self.max_pool(conv(spk), kernel_size=self.pool_size)

                spk, membranes[i+1] = lif(cur, membranes[i+1]) 


            #spk = spk.view(spk.size()[0], -1)
            #spk = nn.Flatten(spk, start_dim=0)
            #print("before flat", spk.size())
            spk = self.flatten(spk)
            #print("after flat", spk.size())
            cur_out = self.fc_out(spk)
            spk_out, membranes[-1] = self.lif_out(cur_out, membranes[-1])

            spk_out_rec.append(spk_out)
            mem_out_rec.append(membranes[-1])

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    
    def forward_ttfs(self, x):
        in_spikes = spikegen.latency(x, num_steps=self.num_steps, linear=True)

        membranes = []
        for layer in self.layers:
            mem = layer.reset_mem() if isinstance(layer, snn.Leaky) else None
            membranes.append(mem)
        #print(type(membranes[-1]))
        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for x_in in in_spikes:
            spk = x_in
            for i in range(0, self.num_conv * 2 - 1, 2):
                conv = self.layers[i]
                lif = self.layers[i+1]

                cur = self.max_pool(conv(spk), kernel_size=self.pool_size)

                #print(i ,type(membranes[i+1]))
                spk, membranes[i+1] = lif(cur, membranes[i+1]) 
                #if spk.sum().item() != 0: print(spk.sum().item())


            spk = self.flatten(spk)

            cur_out = self.fc_out(spk)
            spk_out, membranes[-1] = self.lif_out(cur_out, membranes[-1])

            spk_out_rec.append(spk_out)
            mem_out_rec.append(membranes[-1])
            
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    

def train_csnn(net, optimizer,  train_loader, val_loader, train_config, net_config):
    device, num_epochs, num_steps = train_config['device'],  train_config['num_epochs'], train_config['num_steps']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    val_fn = train_config['val_net']
    loss_hist = []
    val_acc_hist = []
    val_auc_hist = []
    best_auc_roc, best_epoch = 0, 0
    best_net_list = []
    best_val_net = None
    #auc_roc = 0
    val_auc_roc, train_auc_roc = 0, 0
    loss_val = 0
    #print("Epoch:", end ='', flush=True)
    #patience = 30
    #stop_limit = 0
    aux_net = copy.deepcopy(net)
    aux_auc = 0
    for epoch in range(num_epochs):
        net.train()
        #if (epoch + 1) % 10 == 0: print(f"Epoch:{epoch + 1}|auc:{auc_roc}|loss:{loss_val.item()}", flush=True)
        # Minibatch training loop
        for data, targets in train_loader:
            #print(data.size(), data.unsqueeze(1).size())

            data = data.to(device, non_blocking=True).unsqueeze(1)

            targets = targets.to(device, non_blocking=True)
            #print(targets.size(), targets.unsqueeze(1).size())

            # forward pass
            spk_rec, mem_rec = net(data)
            #print(spk_rec, mem_rec)
            #print(mem_rec[:, 0, :])
            # Compute loss
            loss_val = compute_loss(loss_type=loss_type, loss_fn=loss_fn, spk_rec=spk_rec, mem_rec=mem_rec,num_steps=num_steps, targets=targets, dtype=dtype, device=device) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
        _, val_auc_roc = val_fn(net, device, val_loader, train_config)
        if (epoch + 1) % 10 == 0: 
            print(f"Epoch:{epoch + 1}|val_auc:{val_auc_roc:.4f}|loss:{loss_val.item()}", flush=True)

        if val_auc_roc > best_auc_roc:
            best_auc_roc = val_auc_roc
            best_epoch = epoch
            best_val_net = copy.deepcopy(net.state_dict())


    print("Best AUC on val set:", best_auc_roc, "at epoch:", best_epoch)
    return net, loss_hist, val_acc_hist, val_auc_hist, best_net_list, best_val_net


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


def prediction_spk_ttfs_pop(spk_rec):
    spk_times = spk_rec.argmax(dim=0)

    population_c0 = spk_times[:,:spk_rec.shape[2]//2].mean(dim=1)
    population_c1 = spk_times[:,spk_rec.shape[2]//2:].mean(dim=1)

    predicted = torch.stack([population_c0, population_c1], dim=1)
    _, predicted = predicted.min(1).indices
    return predicted


def get_prediction_fn(encoding='rate', pop_coding=False):
    if encoding == 'rate':
        if pop_coding:
            return prediction_spk_rate_pop
        else:
            return prediction_spk_rate
    elif encoding == 'ttfs':
        if pop_coding:
            return prediction_spk_ttfs_pop
        else:
            return prediction_spk_ttfs
    else:
        raise ValueError('Invalid spike prediction fn.')

