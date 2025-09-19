import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import ce_rate_loss, ce_temporal_loss, ce_count_loss
from sklearn.metrics import roc_auc_score, accuracy_score
from snntorch import spikegen
import copy
import numpy as np

# NN Architecture
class SNNet(nn.Module):
    """
    layer_sizes: [#in, #h, #h2.., #out]
    """
    def __init__(self, net_config):
        super().__init__()
        self.num_layers = (len(net_config['layer_sizes']) - 1) * 2
        self.num_steps = net_config['num_steps']
        self.layer_sizes = net_config['layer_sizes']
        self.num_outputs = net_config['layer_sizes'][-1]
        self.encoding = net_config['encoding']
        self.layers = nn.ModuleList()

        for i in range(len(self.layer_sizes) - 1):
            print(self.layer_sizes[i], self.layer_sizes[i+1])
            fc_layer = nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
            self.layers.append(fc_layer)
            lif = snn.Leaky(beta=net_config["beta"], spike_grad=net_config['spike_grad'])
            self.layers.append(lif)

        if self.encoding == "rate":
            self.forward = self.forward_rate
        elif self.encoding == "ttfs":
            self.forward = self.forward_ttfs
        else:
            raise ValueError("Error in encoding type.") 

        
    def forward_rate(self, x):
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
            for i in range(0, self.num_layers, 2):

                fc = self.layers[i]
                lif = self.layers[i+1]

                cur = fc(spk) 

                spk, membranes[i+1] = lif(cur, spk)

            spk_out_rec.append(spk)
            mem_out_rec.append(membranes[-1])

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    
    def forward_ttfs(self, x):
        in_spikes = spikegen.latency(x, num_steps=self.num_steps, linear=True)

        membranes = []
        for layer in self.layers:
            mem = layer.reset_mem() if isinstance(layer, snn.Leaky) else None
            membranes.append(mem)
        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []


        for x_in in in_spikes:
            spk = x_in
            for i in range(0, self.num_layers, 2):

                fc = self.layers[i]
                lif = self.layers[i+1]

                cur = fc(spk) 

                spk, membranes[i+1] = lif(cur, spk)

            spk_out_rec.append(spk)

            mem_out_rec.append(membranes[-1])

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    

def train_snn(net, optimizer,  train_loader, val_loader, train_config, net_config):
    device, num_epochs = train_config['device'],  train_config['num_epochs']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    val_fn = train_config['val_net']
    #scheduler = train_config['scheduler']
    val_acc_hist = []
    val_auc_hist = []
    loss_hist = []
    num_steps = net.num_steps
    best_auc_roc, best_epoch = 0, 0
    best_net_list = []
    best_val_net = None

    val_auc_roc, train_auc_roc = 0, 0
    loss_val = 0

    for epoch in range(num_epochs):
        net.train()
        #print(f"Epoch:{epoch + 1}")

        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # forward pass
            spk_rec, mem_rec = net(data)

            # Compute loss
            loss_val = compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
        #scheduler.step()
        
        val_acc, val_auc_roc = val_fn(net, device, val_loader, train_config)
        if (epoch + 1) % 10 == 0: 
            print(f"Epoch:{epoch + 1}|val_auc:{val_auc_roc:.4f}|loss:{loss_val.item()}", flush=True)

        if val_auc_roc > best_auc_roc:
            best_auc_roc = val_auc_roc
            best_epoch = epoch
            best_val_net = copy.deepcopy(net.state_dict())

    print("Best AUC on val set:", best_auc_roc, "at epoch:", best_epoch)
    return net, loss_hist, val_acc_hist, val_auc_hist, best_net_list, best_val_net


def val_snn(net, device, val_loader, train_config):
    #mean_loss = 0
    val_batch = iter(val_loader)
    accuracy = 0
    auc_roc = 0
    all_preds = []
    all_targets = []
    prediction_fn = train_config['prediction_fn']
    net.eval()
    # Minibatch training loop

    for data, targets in val_batch:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, mem_rec = net(data)

        predicted = prediction_fn(spk_rec)  

        all_preds.extend(predicted.cpu().detach().numpy())
        all_targets.extend(targets.cpu().numpy())

    ## accuracy and roc-auc
    accuracy = accuracy_score(all_targets, np.array(all_preds) >= 0.0)
    auc_roc = roc_auc_score(all_targets, all_preds)
    
    return accuracy, auc_roc


def test_snn(net,  device, test_loader, train_config):
    all_preds = []
    all_targets = []
    prediction_fn = train_config['prediction_fn']
    # Testing Set Loss
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            # forward pass

            test_spk, _ = net(data)

            predicted = prediction_fn(test_spk)        

            all_preds.extend(predicted.cpu().detach().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets

def get_loss_fn(loss_type, class_weights=None, pop_coding=False):
    loss_dict = {
        "rate_loss": ce_rate_loss(weight=class_weights),
        "count_loss": ce_count_loss(weight=class_weights, population_code=pop_coding, num_classes=2),
        "temporal_loss": ce_temporal_loss(weight=class_weights),
        "ce_mem": nn.CrossEntropyLoss(weight=class_weights),
    }

    return loss_dict[loss_type]


def compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device):
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    if loss_type in ["rate_loss", "count_loss"]:
        loss_val = loss_fn(spk_rec, targets)
    elif loss_type == "ce_mem":
        for step in range(num_steps):
            loss_val += loss_fn(mem_rec[step], targets) / num_steps
    elif loss_type == "temporal_loss":
        loss_val = loss_fn(spk_rec, targets)

    return loss_val
