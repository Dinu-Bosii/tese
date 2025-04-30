import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys, Descriptors
import pubchempy as pcp
import torch
from torch.utils.data import random_split, TensorDataset
import torch.nn as nn
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
import numpy as np

from snn_model import SNNet, train_snn, val_snn, test_snn
from csnn_model import CSNNet, train_csnn, val_csnn, test_csnn
from rsnn_model import RSNNet, train_rsnn, val_rsnn, test_rsnn
#from mordred import Calculator, descriptors
import selfies as sf


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


def data_splitter(df, target_name, dataset, split, seed, data_config, dtype):
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
        fp_train, target_train = smile_to_fp(df=train_ids_df, data_config=data_config, target_name=target_name)
        fp_val, target_val = smile_to_fp(df=val_ids_df, data_config=data_config, target_name=target_name)
        fp_test, target_test = smile_to_fp(df=test_ids_df, data_config=data_config, target_name=target_name)

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
    if fp_type == 'count_morgan':
        gen = GetMorganGenerator(radius=radius, fpSize=fp_size, countSimulation=True)
        def fn(mol, **kwargs):
            return gen.GetCountFingerprint(mol, **kwargs)
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


def smile_to_fp(df, data_config, target_name):
    radius = data_config['radius']
    mix = data_config['mix']

    fp1_type, fp1_size = data_config["fp_type"], data_config["num_bits"]
    fp1_gen = fp_generator(fp1_type, fp_size=fp1_size, radius=radius)
    array_size = fp1_size

    if mix:
        fp2_type, fp2_size = data_config["fp_type_2"], data_config['num_bits_2']
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


def smiles_to_descriptor(df, data_config, target_name, missing_val=0):
    num_rows = len(df)
    array_size = len(Descriptors._descList)
    desc_array = np.zeros((num_rows, array_size))
    target_array = np.zeros((num_rows, 1))

    valid_mols = 0
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        
        if mol is None:
            continue

        mol_desc = np.zeros(array_size)
        for k, (nm, fn) in enumerate(Descriptors._descList):
            if k == 42:
                mol_desc[k] = missing_val
                continue
            try:
                val = fn(mol)
                if np.isnan(val):
                    #print(f"NaN in descriptor {nm} for molecule {row['smiles']}")
                    val = missing_val
                elif np.isinf(val):
                    print(f"Inf in descriptor {nm} for molecule {row['smiles']}")
                    val = missing_val
            except Exception as e:
                print(f"Error in descriptor {nm} for {row['smiles']}: {e}")
                val = missing_val
            mol_desc[k] = val

        desc_array[valid_mols]= mol_desc
        target_array[valid_mols] = row[target_name]
        valid_mols += 1

    target_array = target_array.ravel()
    desc_array = desc_array[0:valid_mols]
    target_array = target_array[0:valid_mols]

    return desc_array, target_array


def smiles_to_onehot_selfies(df, data_config, target_name, missing_val=0):
    num_rows = len(df)
    target_array = np.zeros((num_rows, 1))

    valid_mols = 0
    df['selfies'] = df['smiles'].apply(smiles_to_selfies)
    df['selfies'] = df['selfies'].dropna()

    alphabet = sf.get_alphabet_from_selfies(df['selfies'])
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet.add('.')
    alphabet = list(sorted(alphabet))  # ['[=O]', '[C]', '[F]', '[O]', '[nop]']
    print(len(alphabet))
    print(alphabet)
    pad_to_len = max(sf.len_selfies(s) for s in df['selfies'])  # 5
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    selfies_array = np.zeros((num_rows, pad_to_len, pad_to_len))
    for idx, row in df.iterrows():
        try:
            selfies = row['selfies']
            _, one_hot = sf.selfies_to_encoding(
                selfies=selfies,
                vocab_stoi=symbol_to_idx,
                pad_to_len=pad_to_len,
                enc_type="both"
            )

            target_array[valid_mols] = row[target_name]
            valid_mols += 1
        except Exception as e:
            continue


    target_array = target_array.ravel()
    selfies_array = selfies_array[0:valid_mols]
    target_array = target_array[0:valid_mols]

    return selfies_array, target_array


def smiles_to_onehot(df, data_config, target_name, missing_val=0):
    num_rows = len(df)
    target_array = np.zeros((num_rows, 1))
    smiles_list = list(df['smiles'])

    alphabet  = sorted(set("".join(smiles_list)))
    vocab_size = len(alphabet)

    max_len = max(len(s) for s in smiles_list)
    symbol_to_idx = {s: i for i, s in enumerate(list(alphabet))}

    valid_mols = 0
    one_hots = np.zeros((len(smiles_list), max_len, vocab_size), dtype=np.float32)
    for idx, row in df.iterrows():
        try:
            for j, ch in enumerate(row['smiles']):
                    one_hots[idx, j, symbol_to_idx[ch]] = 1.0

            target_array[valid_mols] = row[target_name]
            valid_mols += 1
        except Exception as e:
            continue


    target_array = target_array.ravel()
    one_hots = one_hots[0:valid_mols]
    target_array = target_array[0:valid_mols]

    return one_hots, target_array


def smiles_to_selfies(smiles):
    try:
        selfies = sf.encoder(smiles)
    except sf.EncoderError:
        return None
    except Exception as e:
        return None
    return selfies


def get_spiking_net(net_type, net_config):
    #later on make spike_grad a input parameter
    input_size = net_config["input_size"]
    num_hidden = net_config["num_hidden"]
    time_steps = net_config["num_steps"]
    spike_grad = net_config["spike_grad"]
    beta = net_config["beta"]
    num_outputs = net_config['out_num']
    if net_type == "SNN":
        layer_sizes = [input_size, num_hidden, num_outputs]
        net = SNNet(net_config, layer_sizes=layer_sizes, num_steps=time_steps, spike_grad=spike_grad, beta=beta)
        #num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #print(f"Number of trainable parameters SNN: {num_params}")
        train_fn = train_snn
        val_fn = val_snn
        test_fn = test_snn
        
    elif net_type == "DSNN":
        num_hidden_l2 = net_config["num_hidden_l2"]
        layer_sizes = [input_size, num_hidden, num_hidden_l2, num_outputs]
        net = SNNet(net_config, layer_sizes=layer_sizes, num_steps=time_steps, spike_grad=spike_grad, beta=beta)
        #num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #print(f"Number of trainable parameters DSNN: {num_params}")
        train_fn = train_snn
        val_fn = val_snn
        test_fn = test_snn
        
    elif net_type == "CSNN":
        # Add num_conv parameter if using CSNNet from csnn_model_modular
        #num_conv = net_config['num_conv']
        #net = CSNNet(input_size=input_size, num_steps=time_steps, spike_grad=spike_grad, beta=beta, num_outputs=num_outputs, num_conv=num_conv)
        if net_config["2d"]:
            if isinstance(input_size, int) and input_size == 1024:
                net_config["input_size"]  = [32, 32]
        else:
            if isinstance(input_size, int):
                net_config["input_size"] = [input_size]

        net = CSNNet(net_config)
        train_fn = train_csnn
        val_fn = val_csnn
        test_fn = test_csnn

    elif net_type == "RSNN":
        layer_sizes = [input_size, num_hidden, num_outputs]
        net = RSNNet(layer_sizes=layer_sizes, num_steps=time_steps, spike_grad=spike_grad, beta=beta)
        train_fn = train_rsnn
        val_fn = val_rsnn
        test_fn = test_rsnn
        
    return net, train_fn, val_fn, test_fn


def make_filename(dirname, target, net_type, data_config, lr, wd, optim_type, net_config, train_config, net, model = False):
    results_dir = os.path.join("results", dirname, "")
    if model:
        results_dir = os.path.join(results_dir, "models", "")

    data_str = [] 
    if data_config["repr_type"] == 'descriptor':
        data_str.append("desc")
    elif data_config["repr_type"] == 'fp':
        data_str.append(data_config['fp_type'])
        if data_config['fp_type'] == 'morgan':
            data_str.append(f"r-{data_config['radius']}")
        if data_config['mix']:
            data_str.append(data_config['fp_type_2'])
        if net_config['2d']:
            data_str.append("2D")

    params = [
        None if dirname == 'BBBP' else target, 
        net_type, 
        f"beta-{net_config['beta']}",
        *(
            ["desc"] if data_config["repr_type"] == 'descriptor' else
            (
                [data_config['repr_type']] if data_config["repr_type"] != 'fp' else
                [data_config['fp_type']] + 
                (['r-' + f"{data_config['radius']}"] if data_config['fp_type'] == 'morgan' else []) +
                ([data_config['fp_type_2']] if data_config['mix'] else []) +
                (["2D"] if net_config['2d'] else [])
            )
        ),
        *net_config['input_size'],
        None if net_type == "CSNN" else f"l1{net_config['num_hidden']}",
        None if net_type != "DSNN" else f"l2{net_config['num_hidden_l2']}",
        None if net_type != "CSNN" else "out-" + "-".join(str(layer.out_channels) for layer in net.layers if isinstance(layer, (nn.Conv1d, nn.Conv2d))),
        None if net_type != "CSNN" else f"kernel-{net.conv_kernel}",
        None if net_type != "CSNN" else f"stride-{net.conv_stride}",
        f"t{net_config['num_steps']}",
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


""" def make_filename2(dirname, target, data_config,  net_config, train_config, net, model = False):
    results_dir = os.path.join("results", dirname, "")
    if model:
        results_dir = os.path.join(results_dir, "models", "")
    
    params = []
    
    if dirname != 'BBBP':
        params.append(target)
    if data_config['repr_type'] == "descriptor":
        params.append("desc")
    elif data_config['repr_type'] == "fp":
        params.append(data_config['fp_type'])
        if data_config['fp_type'] == 'morgan':
            params.append(f"r-{data_config['radius']}")
        if data_config['mix']:
            params.append(data_config['fp_type_2'])
    if net_config['2d']:
        params.append("2D")
    
    filename = results_dir + "_".join(str(p) for p in params) + ".csv"
    return filename """