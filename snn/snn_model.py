import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import ce_rate_loss, ce_temporal_loss, ce_count_loss
from sklearn.metrics import roc_auc_score, accuracy_score

# Neurons Count
num_outputs = 2

# Temporal Dynamics
#num_steps = 10
beta = 0.95

# NN Architecture
class SNNet(nn.Module):
    def __init__(self, input_size,num_hidden, num_steps, spike_grad=None, use_l2=False, num_hidden_l2=256):
        super().__init__()
        self.l2 = use_l2
        self.num_steps = num_steps
        if self.l2:
            self.fc1 = nn.Linear(input_size, num_hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc2 = nn.Linear(num_hidden, num_hidden_l2)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc_out = nn.Linear(num_hidden_l2, num_outputs)
            self.lif_out = snn.Leaky(beta=beta, output=True)
        else:
            self.fc1 = nn.Linear(input_size, num_hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc_out = nn.Linear(num_hidden, num_outputs)
            self.lif_out = snn.Leaky(beta=beta, output=True)
        #self.lif2 = snn.Synaptic

    def forward(self, x):
        # Initialize hidden states at t=0

        mem1 = self.lif1.reset_mem()
        if self.l2:
            mem2 = self.lif2.reset_mem()
        mem_out = self.lif_out.reset_mem()
        

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)                  # post-synaptic current <-- spk_in x weight
            spk, mem1 = self.lif1(cur1, mem1) 

            if self.l2:
                cur2 = self.fc2(spk)
                spk, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
    

def train_snn(net, optimizer,  train_loader, val_loader, train_config):
    device, num_epochs = train_config['device'],  train_config['num_epochs']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    val_fn = train_config['val_net']
    val_acc_hist = []
    val_auc_hist = []
    loss_hist = []
    num_steps = net.num_steps
    best_auc_roc = float('inf')
    patience = 6
    for epoch in range(num_epochs):
        net.train()
        print(f"Epoch:{epoch + 1}")
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
            loss_val = compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

        #Naive Early Stopping -> Change to checking for a min_delta for the loss value
        # Do this after every 5th epoch for example

        if epoch % 5 == 0:
            _, auc_roc = val_snn(net, device, val_loader, loss_type, loss_fn, dtype)
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            #val_acc_hist.extend(accuracy)
            val_auc_hist.extend(auc_roc)

    return net, loss_hist, val_acc_hist, val_auc_hist


def val_snn(net, device, eval_loader, loss_type, loss_fn, dtype):
    #mean_loss = 0
    eval_batch = iter(eval_loader)
    accuracy = 0
    auc_roc = 0
    all_preds = []
    all_targets = []

    net.eval()
    # Minibatch training loop
    i = 0
    for data, targets in eval_batch:
        i+=1
        data = data.to(device)
        targets = targets.to(device)

        spk_rec, mem_rec = net(data)
        _, predicted = spk_rec.sum(dim=0).max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        #loss_val = compute_loss(loss_type, loss_fn, spk_rec, mem_rec, targets, dtype, device) 
        # Store loss
        #mean_loss += loss_val
    ## accuracy and roc-auc
    accuracy = accuracy_score(all_targets, predicted)
    auc_roc = roc_auc_score(all_targets, all_preds)
    
    return accuracy, auc_roc


def test_snn(net,  device, test_loader):
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

    print(len(predicted))
    return all_preds, all_targets


def get_loss_fn(loss_type, class_weights=None):
    loss_dict = {
        "rate_loss": ce_rate_loss(weight=class_weights),
        "temporal_loss": ce_temporal_loss(weight=class_weights),
        "cross_entropy": nn.CrossEntropyLoss(weight=class_weights)
    }

    return loss_dict[loss_type]


def compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device):
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    if loss_type == "rate_loss":
        loss_val = loss_fn(spk_rec.to(device), targets.to(device))
    elif loss_type == "cross_entropy":
        for step in range(num_steps):
            loss_val += loss_fn(mem_rec[step].to(device), targets.to(device))
    elif loss_type == "temporal_loss":
        raise ValueError("Temporal Loss is not yet supported.")
    
    return loss_val
