import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys
import torch

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


    return fn
