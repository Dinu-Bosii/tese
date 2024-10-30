import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import ce_count_loss, ce_rate_loss, ce_temporal_loss, ce_count_loss

device = "cpu"
batch_size = 32

# Neurons Count

num_hidden = 1024 #experimentar
num_outputs = 2

# Temporal Dynamics
num_steps = 10
beta = 0.95

# NN Architecture
class Net(nn.Module):
    def __init__(self, num_inputs, spike_grad=None):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, output=True)
        #self.lif2 = snn.Synaptic
    
    #Leaky RELU

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for _ in range(num_steps):
            cur1 = self.fc1(x)                  # post-synaptic current <-- spk_in x weight
            spk1, mem1 = self.lif1(cur1, mem1) 
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    

def train_net(net, optimizer, num_steps, device, num_epochs, train_loader, loss_type, loss_fn, dtype):
    loss_hist = []
    net.train()
    for epoch in range(num_epochs):
        #print(f"Epoch:{epoch}")
    #iter_counter = 0
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

            # Compute loss
            loss_val = compute_loss(loss_type, loss_fn, spk_rec, mem_rec, targets, dtype) 
            #print(loss_val.item())

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

    return net, loss_hist


def test_net(net,  device, test_loader):
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

    return all_preds, all_targets


def get_loss_fn(loss_type, class_weights=None):
    loss_dict = {
        "rate_loss": ce_rate_loss(weight=class_weights),
        "temporal_loss": ce_temporal_loss(weight=class_weights),
        "cross_entropy": nn.CrossEntropyLoss(weight=class_weights)
    }

    return loss_dict[loss_type]


def compute_loss(loss_type, loss_fn, spk_rec, mem_rec, targets, dtype) :
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    if loss_type == "rate_loss":
        loss_val = loss_fn(spk_rec, targets)
    elif loss_type == "cross_entropy":
        for step in range(num_steps):
            loss_val += loss_fn(mem_rec[step], targets)
    elif loss_type == "temporal_loss":
        raise ValueError("Temporal Loss is not yet supported.")
    
    return loss_val
