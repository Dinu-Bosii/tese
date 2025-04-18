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
#from snn_model import SNNet, train_snn, val_snn, test_snn
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
    fp_type, num_bits = fp_config["fp_type"], fp_config["num_bits"]
    radius = fp_config["radius"]
    num_rows = len(df)
    fp_array = np.zeros((num_rows, num_bits))
    target_array = np.zeros((num_rows, 1))
    i = 0

    # Smile to Fingerprint of size {num_bits}
    fp_gen = fp_generator(fp_type, fp_size=num_bits, radius=radius)

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        
        if mol is None:
            continue
        
        if fp_type == "pubchem":
            fingerprint = fp_gen(row['smiles'])
            if fingerprint is None:
                continue
        else:
            fingerprint = fp_gen(mol)

        fp_array[i] = np.array(fingerprint)
        target_array[i] = row[target_name]
        i += 1

    target_array = target_array.ravel()
    fp_array = fp_array[0:i]
    target_array = target_array[0:i]

    return fp_array, target_array

def smile_to_fp_mix(df, fp_config, target_name):
    fp_type, num_bits = fp_config["fp_type"], fp_config["num_bits"]
    fp_type_2, num_bits_2 = fp_config["fp_type_2"], fp_config["num_bits_2"]
    radius = fp_config["radius"]
    num_rows = len(df)
    fp_array = np.zeros((num_rows, num_bits + num_bits_2))
    target_array = np.zeros((num_rows, 1))
    i = 0

    # Smile to Fingerprint of size {num_bits}
    fp_gen = fp_generator(fp_type, fp_size=num_bits, radius=radius)
    fp_gen_2 = fp_generator(fp_type_2, fp_size=num_bits_2, radius=radius)

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        
        if mol is None:
            continue
        

        fingerprint = np.array(fp_gen(mol))
        fingerprint_2 = np.array(fp_gen_2(mol))

        fp_array[i] = np.concatenate([fingerprint, fingerprint_2])
        target_array[i] = row[target_name]
        i += 1

    target_array = target_array.ravel()
    fp_array = fp_array[0:i]
    target_array = target_array[0:i]

    return fp_array, target_array
