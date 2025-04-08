import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys
import pubchempy as pcp
import torch
from torch.utils.data import random_split, TensorDataset
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
import numpy as np
#from csnn_model import CSNNet, train_csnn, val_csnn, test_csnn
from snn_model_modular import SNNet, train_snn, val_snn, test_snn
from csnn_model_modular import CSNNet, train_csnn, val_csnn, test_csnn
#from mordred import Calculator, descriptors


def load_dataset_df(filename):
    file_path = os.path.join('..', 'data', filename)
    df = pd.read_csv(file_path)
    
    targets = []
    if filename == 'tox21.csv':
        targets = df.columns[0:len(df.columns) - 2]
    
    elif filename == 'sider.csv':
        targets = df.columns[1:]
    
    elif filename == 'BBBP.csv':
        targets = [df.columns[2]]

    return df, targets


def data_splitter(df, target_name, dataset, split, seed, fp_config, dtype):
    if split == 'random':
        # Must be a torch dataset
        generator = torch.Generator().manual_seed(int(seed))
        train, val, test = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    elif split == 'scaffold':
        smiles_list = df["smiles"].tolist()
        labels = df[target_name].values

        Xs = np.zeros(len(smiles_list))
        weights = np.zeros(len(smiles_list))

        # Create a DiskDataset
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=labels, w=weights, ids=smiles_list)

        scaffold_splitter = ScaffoldSplitter()
        #by default is 0.8 0.1 0.1
        #print("scaffold splitting..")
        train, val,  test = scaffold_splitter.train_valid_test_split(dataset, seed=seed)

        train_ids_df = pd.DataFrame({'smiles': train.ids, target_name: train.y})
        val_ids_df = pd.DataFrame({'smiles': val.ids, target_name: val.y})
        test_ids_df = pd.DataFrame({'smiles': test.ids, target_name: test.y})
        #print(len(train_ids_df), len(val_ids_df), len(test_ids_df))
        #print("featurizing..")
        fp_train, target_train = smile_to_fp(df=train_ids_df, fp_config=fp_config, target_name=target_name)
        fp_val, target_val = smile_to_fp(df=val_ids_df, fp_config=fp_config, target_name=target_name)
        fp_test, target_test = smile_to_fp(df=test_ids_df, fp_config=fp_config, target_name=target_name)

        fp_train_tensor = torch.tensor(fp_train, dtype=dtype)
        target_train_tensor = torch.tensor(target_train, dtype=dtype).long()
        fp_val_tensor = torch.tensor(fp_val, dtype=dtype)
        target_val_tensor = torch.tensor(target_val, dtype=dtype).long()
        fp_test_tensor = torch.tensor(fp_test, dtype=dtype)
        target_test_tensor = torch.tensor(target_test, dtype=dtype).long()


        train = TensorDataset(fp_train_tensor, target_train_tensor)
        val = TensorDataset(fp_val_tensor, target_val_tensor)
        test = TensorDataset(fp_test_tensor, target_test_tensor)

    return train, val, test


def fp_generator(fp_type, fp_size=1024, radius=2):
    fp_type = fp_type.lower()

    if fp_type == 'morgan':
        gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        def fn(mol, **kwargs):
            return gen.GetFingerprint(mol, **kwargs)

    elif fp_type == 'rdkit':
        gen = GetRDKitFPGenerator(fpSize=fp_size)
        def fn(mol, **kwargs):
            return gen.GetFingerprint(mol, **kwargs)

    elif fp_type == 'maccs':
        def fn(mol, **kwargs):
            return MACCSkeys.GenMACCSKeys(mol, **kwargs)

    elif fp_type == "pubchem": #TODO: this doesn't work and requires internet access
        def pubchem_fp(smiles, **kwargs):
            try:
                compounds = pcp.get_compounds(smiles, 'smiles')
                if not compounds:
                    return None

                pubchem_compound = compounds[0]
            
                fp = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]
            except Exception as e:
                for compound in compounds:
                    print(compound)
            return None
        
        fn = pubchem_fp

        """     
        elif fp_type == 'mordred':
        calc = Calculator(descriptors, ignore_3D=True)

        def mordred_descriptors(mol, **kwargs):
            try:
                result = calc(mol)
                if result.isnull.any():
                    return None  # Handle cases with missing values
                return list(result.values)
            except Exception as e:
                print(f"Error calculating Mordred descriptors: {e}")
                return None

        fn = mordred_descriptors """

    return fn


def smile_to_fp(df, fp_config, target_name):
    radius = fp_config['radius']
    mix = fp_config['mix']

    fp1_type, fp1_size = fp_config["fp_type"], fp_config["num_bits"]
    fp1_gen = fp_generator(fp1_type, fp_size=fp1_size, radius=radius)
    array_size = fp1_size

    if mix:
        fp2_type, fp2_size = fp_config["fp_type_2"], fp_config['num_bits_2']
        fp2_gen = fp_generator(fp2_type, fp_size=fp2_size, radius=radius)
        array_size += fp2_size

    num_rows = len(df)

    fp_array = np.zeros((num_rows, array_size))
    target_array = np.zeros((num_rows, 1))

    valid_mols = 0
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        
        if mol is None:
            continue

        # Additional checks may be required for Pubchem fp
        fingerprint = np.array(fp1_gen(mol))
        if mix:
            fingerprint_2 = np.array(fp2_gen(mol))
            fingerprint = np.concatenate([fingerprint, fingerprint_2])


        fp_array[valid_mols] = fingerprint
        target_array[valid_mols] = row[target_name]
        valid_mols += 1

    target_array = target_array.ravel()
    fp_array = fp_array[0:valid_mols]
    target_array = target_array[0:valid_mols]

    return fp_array, target_array


def get_spiking_net(net_type, net_config):
    #later on make spike_grad a input parameter
    input_size = net_config["input_size"]
    num_hidden = net_config["num_hidden"]
    time_steps = net_config["time_steps"]
    spike_grad = net_config["spike_grad"]
    beta = net_config["beta"]
    num_outputs = net_config['out_num']
    if net_type == "SNN":
        layer_sizes = [input_size, num_hidden, num_outputs]
        net = SNNet(layer_sizes=layer_sizes, num_steps=time_steps, spike_grad=spike_grad, beta=beta)
        #num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #print(f"Number of trainable parameters SNN: {num_params}")
        train_fn = train_snn
        val_fn = val_snn
        test_fn = test_snn
        
    elif net_type == "DSNN":
        num_hidden_l2 = net_config["num_hidden_l2"]
        layer_sizes = [input_size, num_hidden, num_hidden_l2, num_outputs]
        net = SNNet(layer_sizes=layer_sizes, num_steps=time_steps, spike_grad=spike_grad, beta=beta)
        #num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #print(f"Number of trainable parameters DSNN: {num_params}")
        train_fn = train_snn
        val_fn = val_snn
        test_fn = test_snn
        
    elif net_type == "CSNN":
        # Add num_conv parameter if using CSNNet from csnn_model_modular
        #num_conv = net_config['num_conv']
        #net = CSNNet(input_size=input_size, num_steps=time_steps, spike_grad=spike_grad, beta=beta, num_outputs=num_outputs, num_conv=num_conv)
        if net_config["2d"] and input_size == 1024:
            input_size = [32, 32]
        else:
            input_size = [input_size]
        net = CSNNet(input_size=input_size, num_steps=time_steps, spike_grad=spike_grad, beta=beta, num_outputs=num_outputs)
        train_fn = train_csnn
        val_fn = val_csnn
        test_fn = test_csnn

    return net, train_fn, val_fn, test_fn


def make_filename(dirname, target, net_type, fp_config, lr, wd, optim_type, net_config, train_config, net, model = False):
    results_dir = os.path.join("results", dirname, "")
    if model:
        results_dir = os.path.join(results_dir, "models", "")

    params = [
        None if dirname == 'BBBP' else target, 
        net_type, 
        f"beta-{net_config['beta']}",
        fp_config['fp_type'],
        None if fp_config['fp_type'] != 'morgan' else 'r-' + f"{fp_config['radius']}",
        fp_config['fp_type_2'] if fp_config['mix'] else None,
        None if not net_config['2d'] else "2D",
        net_config['input_size'],
        None if net_type == "CSNN" else f"l1{net_config['num_hidden']}",
        None if net_type != "DSNN" else f"l2{net_config['num_hidden_l2']}",
        None if net_type != "CSNN" else f"out-{net.conv1.out_channels}" + (f"-{net.conv2.out_channels}" if hasattr(net, "conv2") else ""),
        None if net_type != "CSNN" else f"kernel-{net.conv_kernel}",
        None if net_type != "CSNN" else f"stride-{net.conv_stride}",
        f"t{net_config['time_steps']}",
        f"e{train_config['num_epochs']}",
        f"b{train_config['batch_size']}",
        f"lr{lr}",
        train_config['loss_type'],
        optim_type,
        f"wd{wd}",
        None if net_config['spike_grad'] is None else f"sig-{net_config['slope']}",
        "no-bias" if not net_config['bias'] else "bias",
        None if net_config['out_num'] == 2 else f"pop-{net_config['out_num']}",
    ]

    filename = results_dir + "_".join(str(p) for p in params if p is not None) + ".csv"
    return filename

