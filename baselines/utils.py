import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys, Descriptors
import pubchempy as pcp
import torch
from torch.utils.data import random_split, TensorDataset
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
import numpy as np
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
