import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import ce_rate_loss, ce_temporal_loss, ce_count_loss
from sklearn.metrics import roc_auc_score, accuracy_score


# NN Architecture
class RSNNet(nn.Module):
    """
    layer sizes = [#in, #h, #h2.., #out]
    """
    def __init__(self, layer_sizes, num_steps, spike_grad=None, beta=0.95):
        super().__init__()
        self.num_layers = len(layer_sizes)
        self.num_steps = num_steps
        self.layer_sizes = layer_sizes
        self.num_outputs = layer_sizes[-1]
        self.layers = nn.ModuleList()

        for i in range(self.num_layers - 1):
            fc_layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(fc_layer)
            lif = snn.RLeaky(beta=beta, spike_grad=spike_grad, all_to_all=False)
            self.layers.append(lif)

        #print(self.layers)
    def forward(self, x):
        # The neuron type should be dynamic as well
        
        # Initialize hidden states at t=0
        membranes = [layer.reset_mem() for layer in self.layers[1::2]]
        
        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []
        #print("num layers:", self.num_layers)
        for _ in range(self.num_steps):
            spk = x

            #cur_1 = self.layers[0](spk)


            for i in range(0, self.num_layers, 2):
                fc = self.layers[i]
                lif = self.layers[i+1]
              
                #if i == 0:
                #    cur = cur_1
                #else:
                #    cur = fc(spk) 
                cur = fc(spk) 

                spk, membranes[i//2] = lif(cur, spk)

            spk_out_rec.append(spk)
            mem_out_rec.append(membranes[-1])
        #print(torch.stack(spk_out_rec, dim=0).size())
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    

def train_rsnn(net, optimizer,  train_loader, val_loader, train_config, net_config):
    device, num_epochs = train_config['device'],  train_config['num_epochs']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    val_fn = train_config['val_net']
    #scheduler = train_config['scheduler']
    val_acc_hist = []
    val_auc_hist = []
    loss_hist = []
    num_steps = net.num_steps
    best_auc_roc = 0
    #patience = 10
    #patience_counter = 0
    best_net_list = []
    #epoch_list = []
    auc_roc = 0
    loss_value = 0
    print("Epoch:0", end ='', flush=True)
    for epoch in range(num_epochs):
        net.train()
        if (epoch + 1) % 10 == 0: print(f"Epoch:{epoch + 1}|auc:{auc_roc}|loss:{loss_value.item()}", flush=True)

        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)
            #print(data.size(), data.view(batch_size, -1).size())
            #print(data.shape)  # Should be [32, 1, 167]
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)
            #print(spk_rec, mem_rec)
            #print(spk_rec.size(), mem_rec.size(), targets.size())
            # Compute loss
            loss_value = compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            # Store loss history for future plotting
            loss_hist.append(loss_value.item())
        #scheduler.step()
        
        _, auc_roc = val_fn(net, device, val_loader, train_config)
        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc   
        
        best_net_list.append(net.state_dict())

            #val_acc_hist.extend(accuracy)
        val_auc_hist.extend([auc_roc])

    return net, loss_hist, val_acc_hist, val_auc_hist, best_net_list#, epoch_list


def val_rsnn(net, device, val_loader, train_config):
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
        #print(spk_rec.size())

        predicted = prediction_fn(spk_rec)  

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        #loss_val = compute_loss(loss_type, loss_fn, spk_rec, mem_rec, targets, dtype, device) 
        # Store loss
        #mean_loss += loss_val
    ## accuracy and roc-auc
    accuracy = accuracy_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)
    
    return accuracy, auc_roc


def test_rsnn(net,  device, test_loader, train_config):
    all_preds = []
    all_targets = []
    #all_probs = []
    prediction_fn = train_config['prediction_fn']
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

            predicted = prediction_fn(test_spk)        

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets

# ver binary cross entropy, bcewithlogits
def get_loss_fn(loss_type, class_weights=None, pop_coding=False):
    loss_dict = {
        "rate_loss": ce_rate_loss(weight=class_weights),
        "count_loss": ce_count_loss(weight=class_weights, population_code=pop_coding, num_classes=2),
        "temporal_loss": ce_temporal_loss(weight=class_weights),
        "ce_mem": nn.CrossEntropyLoss(weight=class_weights),
        "bce_loss": nn.BCEWithLogitsLoss(weight=class_weights[1])
    }

    return loss_dict[loss_type]


def compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device):
    loss_val = torch.zeros((1), dtype=dtype, device=device)
    #targets = targets.unsqueeze(1)
    #print(mem_rec[0].size(), targets.size())
    #print(spk_rec[0].size(), targets.size())

    if loss_type in ["rate_loss", "count_loss"]:
        loss_val = loss_fn(spk_rec, targets)
    elif loss_type in ["ce_mem", "bce_loss"]:
        for step in range(num_steps):
            loss_val += loss_fn(mem_rec[step], targets) / num_steps
    elif loss_type == "temporal_loss":
        loss_val = loss_fn(spk_rec, targets)

    return loss_val
