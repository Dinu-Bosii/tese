import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys
import pubchempy as pcp
import torch
from torch.utils.data import random_split
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
import numpy as np

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

def data_splitter(df, target, split, seed):
    if split == 'random':
        generator = torch.Generator().manual_seed(seed)
        train, val, test = random_split(df, [0.8, 0.1, 0.1], generator=generator)

    elif split == 'scaffold':
        smiles_list = df["smiles"].tolist()
        labels = df[target].values

        Xs = np.zeros(len(smiles_list))
        weights = np.zeros(len(smiles_list))

        # Create a DiskDataset
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=labels, w=weights, ids=smiles_list)

        scaffold_splitter = ScaffoldSplitter()
        train, valid,  test = scaffold_splitter.train_valid_test_split(dataset, seed=seed)
    return train, val, test


def fp_generator(fp_type, fp_size=1024, radius=2):
    fp_type = fp_type.lower()

    if fp_type == 'morgan':
        gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        fn = lambda mol, **kwargs: gen.GetFingerprint(mol, **kwargs)

    elif fp_type == 'rdkit':
        gen = GetRDKitFPGenerator(fpSize=fp_size)
        fn = lambda mol, **kwargs: gen.GetFingerprint(mol, **kwargs)

    elif fp_type == 'maccs':
        fn = lambda mol, **kwargs: MACCSkeys.GenMACCSKeys(mol, **kwargs)

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

    return fn


""" from mordred import Calculator, descriptors
mol = Chem.MolFromSmiles("CCO")

# Initialize Mordred descriptor calculator
calc = Calculator(descriptors)

# Calculate descriptors for the molecule
result = calc(mol)

# Print descriptor names and values
for name, value in result.items():
    print(value) """

def bbbp_splitter(df):
    pass
