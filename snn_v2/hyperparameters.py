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
import matplotlib.pyplot as plt
from utils import load_dataset_df, smile_to_fp,smiles_to_descriptor,smiles_to_onehot,smiles_to_onehot_selfies,data_splitter,get_spiking_net,make_filename
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score
from csnn_model import get_prediction_fn
from snntorch import surrogate


# In[2]:


import sys
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"./results/logs/output_{timestamp}.txt"
log_file = open(log_path, "w")
sys.stdout = log_file


# #### Load DataFrame

# In[3]:


files = ['tox21.csv','sider.csv', 'BBBP.csv']
dt_file = files[1]
dirname = dt_file.removesuffix('.csv')

df, targets = load_dataset_df(filename=dt_file)

for t in targets:
    df_temp = df[[t, 'smiles']].dropna()
    class_counts = df[t].count()
    class_sum = df[t].sum()
    print(t, class_counts, round(class_sum/class_counts, 2)) 


# In[4]:


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


# #### Molecular Representation

# In[5]:


representations = ["fp", "descriptor", "SELFIES-1hot", "SMILES-1hot"]#, "graph-list"]

repr_type = representations[0]


# In[6]:


if repr_type == "fp":
    fp_types = [['morgan', 1024], ['maccs', 167], ['RDKit', 1024], ['count_morgan', 1024], ['pubchem', 881]]
    mix = False
    fp_type, num_bits = fp_types[0]
    if mix and fp_type == 'RDKit':
        num_bits = 512
    data_config = {"fp_type": fp_type,
                "num_bits": num_bits,
                "radius": 2,
                "fp_type_2": fp_types[0][0],
                "num_bits_2": 1024 - num_bits,
                "mix": mix,}
    dim_2 = False
    print(fp_type, '-', num_bits)
    if mix: print(data_config['fp_type_2'], '-', data_config['num_bits_2'])
    if dim_2: print("2D FP")

elif repr_type == "descriptor":
    desc_type = ["RDKit", "TODO"]
    data_config = {"desc": desc_type[0],
                   "size": 0,
                }
elif repr_type == "SELFIES-1hot":
    dim_2 = True
    data_config = {}

elif repr_type == "SMILES-1hot":
    dim_2 = True
    data_config = {}

data_config["repr_type"] = repr_type
print(repr_type)


# In[7]:


dtype = torch.float
split = "scaffold"
dataset = None

if dirname != 'BBBP':
    split = "random"
    if repr_type == "fp":
        fp_array, target_array = smile_to_fp(df, data_config=data_config, target_name=target_name)
        # Create Torch Dataset
        fp_tensor = torch.tensor(fp_array, dtype=dtype)
        print(fp_tensor.size())
        target_tensor = torch.tensor(target_array, dtype=dtype).long()
        if dim_2:
            fp_tensor = fp_tensor.view(-1, 32, 32)
            print(fp_tensor.size())
        dataset = TensorDataset(fp_tensor, target_tensor)
    elif repr_type == "descriptor":
        desc_array, target_array = smiles_to_descriptor(df, data_config=data_config, target_name=target_name, missing_val=0)
        # Create Torch Dataset
        desc_tensor = torch.tensor(desc_array, dtype=dtype)
        target_tensor = torch.tensor(target_array, dtype=dtype).long()

        dataset = TensorDataset(desc_tensor, target_tensor)
        print(desc_tensor.size())
    elif repr_type == "SELFIES-1hot":
        selfies_array, target_array = smiles_to_onehot_selfies(df, data_config=data_config, target_name=target_name, missing_val=0)
        # Create Torch Dataset
        selfies_tensor = torch.tensor(selfies_array, dtype=dtype)
        target_tensor = torch.tensor(target_array, dtype=dtype).long()

        dataset = TensorDataset(selfies_tensor, target_tensor)
        print(selfies_tensor.size())
    elif repr_type == "SMILES-1hot":
        smiles_array, target_array = smiles_to_onehot(df, data_config=data_config, target_name=target_name, missing_val=0)
        # Create Torch Dataset
        smiles_tensor = torch.tensor(smiles_array, dtype=dtype)
        target_tensor = torch.tensor(target_array, dtype=dtype).long()

        dataset = TensorDataset(smiles_tensor, target_tensor)
        print(smiles_tensor.size())


# In[8]:


if repr_type == "SMILES_1hot":
    longest_smiles = df.loc[df['smiles'].str.len().idxmax(), 'smiles']
    print(longest_smiles)

    from rdkit import Chem

    mol = Chem.MolFromSmiles(longest_smiles)
    print("Valid" if mol else "Invalid")
    cc = df[df['smiles'].str.len() > 256]

    print(len(cc))
    sample1 = smiles_array[0]
    print(sample1)
    print(selfies_tensor.size())


# In[9]:


if repr_type == "descriptor":
    from rdkit.Chem import  Descriptors
    print("desc_array Has NaNs:", np.isnan(desc_array).any())
    print("desc_array Has Infs:", np.isinf(desc_array).any())
    print("desc_tensor has nans:", torch.isnan(desc_tensor).any().item())
    print("desc_tensor has infs:", torch.isinf(desc_tensor).any().item())

    print("Max value in desc_array:", np.max(desc_array))

    # Find the index of the max value in the array
    max_idx = np.argmax(desc_array)  # Returns the index of the max value in flattened array

    # Find the corresponding row and descriptor index
    row_idx = max_idx // desc_array.shape[1]  # Row index (which molecule)
    desc_idx = max_idx % desc_array.shape[1]  # Descriptor index (which descriptor)
    print(f"Max value at row {row_idx}, descriptor {desc_idx} with value: {desc_array[row_idx, desc_idx]}")

    for k, (nm, fn) in enumerate(Descriptors._descList):
        print(k, nm)


# #### Loss Function

# In[ ]:


from sklearn.utils.class_weight import compute_class_weight

loss_types = ['ce_mem', 'rate_loss', 'count_loss', 'temporal_loss', 'bce_loss']
loss_type = loss_types[2]
print(loss_type)


# #### Training

# In[11]:


net_types = ["SNN", "DSNN", "CSNN", "RSNN"]
net_type = net_types[2]
slope = 10
spike_grad = surrogate.fast_sigmoid(slope=slope)
#spike_grad = None
beta = 0.95 
bias = True
net_config = {
            "num_hidden": 1024,
            "num_hidden_l2": 256,
            "num_steps": 10,
            "spike_grad": spike_grad,
            "slope": None if not spike_grad else slope, #spike_grad.__closure__[0].cell_contents,
            "beta": beta,
            "encoding": 'rate' if loss_type != 'temporal_loss' else 'ttfs',
            "bias": bias,
            "out_num": 2
            }
#print(spike_grad.__closure__[0].cell_contents)
if net_type == "CSNN":
    net_config['num_conv'] = 1
    net_config['conv_stride'] = [1 for _ in range(net_config['num_conv'])]
    net_config["pool_size"] = 2
    net_config["conv_kernel"] = 3
    #net_config["conv_stride"] = 1
    net_config["conv_groups"] = 1

if repr_type == "fp":
    net_config["input_size"] = 1024 if data_config['mix'] else num_bits
    net_config["2d"] = dim_2

elif repr_type == "descriptor":
    net_config["input_size"] = desc_tensor.shape[1]
    net_config["2d"] = False
    net_config["time_steps"] = 10

if repr_type == "SELFIES-1hot":
    net_config["input_size"] = [desc_tensor.shape[1],desc_tensor.shape[2]] 
    net_config["2d"] = True
if repr_type == "SMILES-1hot":
    net_config["2d"] = True
    net_config["input_size"] = [desc_tensor.shape[1],desc_tensor.shape[2]] 
print(net_type)


# In[ ]:


pop_coding = net_config['out_num'] > 2
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
                'num_steps': net_config['num_steps'],
                'val_net': None,
                'prediction_fn': get_prediction_fn(encoding=net_config['encoding'], pop_coding=pop_coding),
                }
drop_last = net_type == "CSNN"
pin_memory = device == "cuda"
save_csv = True
save_models = True
results = [[], [], [], [], [], []]


# In[13]:


print("-----Configuration-----")
print(data_config)
print(net_config)
print(train_config)


# In[14]:


from rdkit import RDLogger

# Disable RDKit logging for the scaffold meeting
RDLogger.DisableLog('rdApp.*')


# In[15]:


def calc_metrics(metrics_list, all_targets, all_preds):
    accuracy = accuracy_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    
    metrics_list[0].append(accuracy)
    metrics_list[1].append(auc_roc)
    metrics_list[2].append(sensitivity)
    metrics_list[3].append(specificity)
    metrics_list[4].append(f1)
    metrics_list[5].append(precision)
    


# In[16]:


import itertools
from collections import OrderedDict

# for CSNN
if net_type == "CSNN":
    search_space = OrderedDict({
        "bias": [True, False],
        "beta": [0.9, 0.7, 0.5],
        "learning_rate": [1e-3, 1e-4, 1e-5],
        "out_num": [2, 10, 20, 50],
        "spike_grad": [None, surrogate.fast_sigmoid(slope=10), surrogate.fast_sigmoid(slope=25), surrogate.fast_sigmoid(slope=50)],
        "conv_groups": [1, 2],
        "conv_kernel": [3, 5, 7],
        "pool_size": [2, 4],
        "conv_stride": [1, 2],
        #"loss_fn": ['ce_mem', 'rate_loss', 'count_loss'],
        "num_steps": [5, 10, 20, 50]
    })
    net_config = {
        "beta": 0.9,
        "spike_grad": None,
        "slope": slope,
        "encoding": 'rate',
        "out_num": 2,
        'num_conv': 1,
        #"out_num": params['out_num'],
        #"learning_rate": 1e-4,
        #"pool_size" : 2,
        #"conv_kernel": params["conv_kernel"],
        #"conv_stride": 1,
        #"conv_groups": params["conv_groups"],
        "2d": dim_2,
    }
else:
# for feedforward SNN
    search_space = OrderedDict({
        #"num_hidden": [512, 1024, 2048],
        "num_hidden": [1024, 2048],
        #"num_hidden_l2": [1024, 512],
        #"num_layers": [1, 2],
        "beta": [0.9, 0.7, 0.5],
        "spike_grad": [None, surrogate.fast_sigmoid(slope=10), surrogate.fast_sigmoid(slope=25), surrogate.fast_sigmoid(slope=50)],
        "bias": [True, False],
        "learning_rate": [1e-3, 1e-4, 1e-5],
        "out_num": [2, 10, 20, 50],
        #"loss_type": ['ce_mem', 'rate_loss', 'count_loss'],
        "num_steps": [5, 10, 20, 50]
    })
    net_config = {
        "beta": 0.9,
        "spike_grad": None,
        "num_hidden": 1024,
        "num_steps": 10,
        "slope": slope,
        "encoding": 'rate',
        "out_num": 2,
        "2d": dim_2,
        "learning_rate": 1e-4,
    }


# In[ ]:


import time

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
keys = list(search_space.keys())
combinations = list(itertools.product(*search_space.values()))
random.shuffle(combinations)
combinations = combinations[:200]


# In[18]:


net_params = ["bias", "beta", "learning_rate", "out_num", "num_steps", "num_hidden", "num_hidden_l2"]
train_params = ["num_steps", "learning_rate"], #"loss_type"]
print(search_space)


# In[ ]:


for i, values in enumerate(combinations):
    params = dict(zip(keys, values))
    print(f"\n=== Trial {i + 1}/{len(combinations)} ===")
    #print("Params:", params)
    for key, value in params.items():
        if key =='spike_grad':
            if value is not None:
                slope = value.__closure__[0].cell_contents
                print('spike_grad: fast_sigmoid -', slope, flush=True)
                net_config['slope'] = slope
            else:
                print('spike_grad: arctan', flush=True)
        else:
            print(key, ':', value, flush=True)
        if key in net_params:
            net_config[key] = value
        if key in train_params:
            train_config[key] = value

    net_config["input_size"] = 1024 if data_config['mix'] else num_bits

    if net_type == "CSNN":
        #net_config['conv_stride'] = [1 for _ in range(net_config['num_conv'])]
        net_config['conv_stride'] = [params['conv_stride'] for _ in range(net_config['num_conv'])]
        net_config['conv_kernel'] = params["conv_kernel"]
        net_config['conv_groups'] = params["conv_groups"]
        net_config['pool_size'] = params["pool_size"]
    
    pop_coding = net_config['out_num'] > 2
    train_config['loss_type'] = loss_type
    train_config['prediction_fn'] = get_prediction_fn(encoding=net_config['encoding'], pop_coding=pop_coding)

    net, train_net, val_net, test_net = get_spiking_net(net_type, net_config)
    net = net.to(device)
    train_config['val_net'] = val_net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    train_config["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['num_epochs'])
    
    # DATA SPLIT
    train, val, test = data_splitter(df, target_name, split=split, dataset=dataset, data_config=data_config, seed=seed, dtype=dtype)
    _, train_label = train[:]
    _, val_label = val[:]
    _, test_label = test[:]
        

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # LOSS FN
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1], dtype=np.int8), y=np.array(train_label, dtype=np.int8))
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    train_config["loss_fn"] = get_loss_fn(loss_type=train_config["loss_type"], class_weights=class_weights, pop_coding=pop_coding)


    # TRAINING
    start_time = time.time()
    net, loss_hist, val_acc_hist, val_auc_hist, net_list, best_val_net = train_net(net=net, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, train_config=train_config, net_config=net_config)
    end_time = time.time()
    train_time = end_time - start_time
    print()
    print(f"Time: {train_time:.4f} seconds")
    all_preds, all_targets = test_net(net, device, test_loader, train_config)
    auc_roc_test = roc_auc_score(all_targets, all_preds)
    print('Last model AUC on test set:', auc_roc_test)
    model = net
    model.load_state_dict(best_val_net)
    all_preds, all_targets = test_net(model, device, test_loader, train_config)
    auc_roc_test = roc_auc_score(all_targets, all_preds)
    print('Best model AUC on test set:', auc_roc_test)

    """     model = net
    ensemble_preds =  np.zeros_like(all_preds)   
    print("Ensemble models:")
    for state_dict in net_list:
        model.load_state_dict(state_dict)
        all_preds, _ = test_net(net, device, test_loader, train_config)
        auc_roc_test = roc_auc_score(all_targets, all_preds)
        print("....AUC:",auc_roc_test)
        ensemble_preds += all_preds
    ensemble_preds = (ensemble_preds >= 3).astype(int)
    auc_roc_test_ensemble = roc_auc_score(all_targets, ensemble_preds)
    print('ensemble AUC on test set:', auc_roc_test_ensemble)
    """

    result_entry = {
        "params": params,
        "auc_test": auc_roc_test,
    }
    results.append(result_entry)


# In[ ]:


param_results = results[6:]
param_results.sort(key=lambda x: x["auc_test"], reverse=True)
print("\nTop Configs:")
for r in param_results:
    print(r)

