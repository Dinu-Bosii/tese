#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from rdkit import Chem
from snn_model import get_loss_fn
import torch
from torch.utils.data import TensorDataset, DataLoader
from snntorch import spikegen, surrogate
import matplotlib.pyplot as plt
from utils import load_dataset_df, smile_to_fp, data_splitter, get_spiking_net, make_filename
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score
from csnn_model import CSNNet, get_prediction_fn, bias


# #### Load DataFrame

# In[4]:


files = ['tox21.csv','sider.csv', 'BBBP.csv']
dt_file = files[0]
dirname = dt_file.removesuffix('.csv')

df, targets = load_dataset_df(filename=dt_file)

for t in targets:
    df_temp = df[[t, 'smiles']].dropna()
    class_counts = df[t].count()
    class_sum = df[t].sum()
    print(t, class_counts, round(class_sum/class_counts, 2)) 


# In[5]:


if dirname == 'tox21':
    # SR-ARE
    target_name = targets[7]
    # SR-MMP
elif dirname == 'sider':
    #Hepatobiliary disorders 1427 samples, 0.52 class ratio
    target_name = targets[0]
else:
    target_name = targets[0]
    
df = df[[target_name, 'smiles']].dropna()


# #### SMILE to Fingerprint

# In[6]:


fp_types = [['morgan', 1024], ['maccs', 167], ['RDKit', 1024], ['pubchem', 881]]
mix = True
fp_type, num_bits = fp_types[2]
if mix and fp_type == 'RDKit':
    num_bits = 512
fp_config = {"fp_type": fp_type,
             "num_bits": num_bits,
             "radius": 2,
             "fp_type_2": fp_types[0][0],
             "num_bits_2": 1024 - num_bits,
             "mix": mix,
             }

print(fp_type, '-', num_bits)
if mix:
   print(fp_config['fp_type_2'], '-', fp_config['num_bits_2']) 


# In[8]:


dtype = torch.float32
split = "scaffold"
dataset = None

if dirname != 'BBBP':
    split = "random"

    fp_array, target_array = smile_to_fp(df, fp_config=fp_config, target_name=target_name)
    # Create Torch Dataset
    fp_tensor = torch.tensor(fp_array, dtype=dtype)
    target_tensor = torch.tensor(target_array, dtype=dtype).long()

    dataset = TensorDataset(fp_tensor, target_tensor)


# In[ ]:


print(fp_array.shape)


# #### Loss Function

# In[ ]:


from sklearn.utils.class_weight import compute_class_weight

loss_types = ['ce_mem', 'rate_loss', 'count_loss', 'temporal_loss', 'bce_loss']
loss_type = loss_types[2]
print(loss_type)


# #### Train Loop

# In[ ]:


net_types = ["SNN", "DSNN", "CSNN"]
net_type = net_types[2]
slope = 10
#spike_grad = surrogate.fast_sigmoid(slope=slope)
spike_grad = None
beta = 0.95 #experimentar 0.7

net_config = {"input_size": 1024 if fp_config['mix'] else num_bits,
              "num_hidden": 512,
              "num_hidden_l2": 256,
              "use_l2": net_type == "DSNN",
              "time_steps": 10,
              "spike_grad": spike_grad,
              "slope": None if not spike_grad else slope, #spike_grad.__closure__[0].cell_contents,
              "beta": beta,
              "encoding": 'rate',
              "bias": bias,
              "out_num": 2
              }
pop_coding = net_config['out_num'] > 2


# In[ ]:


lr=1e-4 #1e-6 default for 1000 epochs. csnn requires higher
iterations = 30
weight_decay = 0 # 1e-5
optim_type = 'Adam'
#optim_type = 'SGD'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 16 #16, 8
train_config = {"num_epochs": 1000,
                "batch_size": batch_size,
                "device": device,
                "loss_type": loss_type,
                "loss_fn": None,
                'dtype': dtype,
                'num_steps': net_config['time_steps'],
                'val_net': None,
                'prediction_fn': get_prediction_fn(encoding=net_config['encoding'], pop_coding=pop_coding),
                }
drop_last = net_type == "CSNN"
pin_memory = device == "cuda"
save = True
results = [[], [], [], [], [], []]


# In[ ]:


print("-----Configuration-----")
print(net_config)
print(train_config)


# In[ ]:


from rdkit import RDLogger

# Disable RDKit logging for the scaffold meeting
RDLogger.DisableLog('rdApp.*')


# In[ ]:


def calc_metrics(metrics_list, all_targets, all_preds):

    accuracy = accuracy_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    
    print(accuracy, auc_roc)
    metrics_list[0].append(accuracy)
    metrics_list[1].append(auc_roc)
    metrics_list[2].append(sensitivity)
    metrics_list[3].append(specificity)
    metrics_list[4].append(f1)
    metrics_list[5].append(precision)
    


# In[ ]:


# The worst and best iterations (with maccs only): 16, 29


# In[ ]:


for iter in range(iterations):
    print(f"Iteration:{iter + 1}/{iterations}")
    seed = iter + 1
    print(f"Seed:{seed}")
    random.seed(seed)

    net, train_net, val_net, test_net = get_spiking_net(net_type, net_config)
    net = net.to(device)
    train_config['val_net'] = val_net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    #optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['num_epochs'])
    #optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #train_config["scheduler"] = scheduler

    # DATA SPLIT
    train, val, test = data_splitter(df, target_name, split=split, dataset=dataset, fp_config=fp_config, seed=seed, dtype=dtype)
    _, train_label = train[:]
    _, val_label = val[:]
    _, test_label = test[:]
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # LOSS FN
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1], dtype=np.int8), y=np.array(train_label, dtype=np.int8))
    #class_weights[0] = class_weights[0]/2 
    #class_weights[0] = class_weights[0]*2

    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    train_config["loss_fn"] = get_loss_fn(loss_type=loss_type, class_weights=class_weights, pop_coding=pop_coding)


    # TRAINING
    net, loss_hist, val_acc_hist, val_auc_hist, net_list = train_net(net=net, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, train_config=train_config, net_config=net_config)
    
    # TESTING
    model = net
    best_test_auc = 0
    best_epoch = 0
    for index, model_dict in enumerate(net_list):
        model.load_state_dict(model_dict)
        model.to(device)
        all_preds2, all_targets2 = test_net(model, device, test_loader, train_config)
        auc_roc_test = roc_auc_score(all_targets2, all_preds2)
        if auc_roc_test > best_test_auc:
            best_test_auc, best_epoch = (auc_roc_test, index)

    print(best_epoch, best_test_auc)
    model.load_state_dict(net_list[best_epoch])
    filename = make_filename(dirname, target_name, net_type, fp_config, lr, weight_decay, optim_type, net_config, train_config, model, model = True)
    model_name = filename.removesuffix('.csv') + f"seed-{seed}" +'.pth'
    torch.save(model.state_dict(), model_name)
    all_preds, all_targets = test_net(model, device, test_loader, train_config)
    calc_metrics(results, all_preds=all_preds, all_targets=all_targets)


# #### Save Metrics

# In[ ]:


metrics_np = np.zeros(12)

for i, metric in enumerate(results):
    metrics_np[i*2] = np.round(np.mean(metric), 3)
    metrics_np[i*2+1] = np.round(np.std(metric), 3)

# Print Results
print(f"Accuracy:  {metrics_np[0]:.3f} ± {metrics_np[1]:.3f}")
print(f"AUC ROC: {metrics_np[2]:.3f} ± {metrics_np[3]:.3f}")
print(f"Sensitivity: {metrics_np[4]:.3f} ± {metrics_np[5]:.3f}")
print(f"Specificity: {metrics_np[6]:.3f} ± {metrics_np[7]:.3f}")

metric_names = ['Acc', 'AUC', 'Sn', 'Sp', 'F1', 'Precision']
metrics_np = metrics_np.reshape(1, -1)
columns = []
for name in metric_names:
    columns.extend([f'Mean {name}', f'Std {name}'])


df_metrics = pd.DataFrame(metrics_np, columns=columns)
num_hidden = net_config['num_hidden']
time_steps = train_config['num_steps']
num_epochs = train_config['num_epochs']

# TODO: Add neuron thresholds to name
filename = make_filename(dirname, target_name, net_type, fp_config, lr, weight_decay, optim_type, net_config, train_config, model)
if save: df_metrics.to_csv(filename, index=False)

print(filename)


