import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.utils import from_smiles 
from utils.utils import SmilesEnumerator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

class SmileDataset(Dataset):

    def __init__(self, debug, data, content, block_size, 
                 aug_prob = 0.5,  scaffold = None, 
                 scaffold_maxlen = None):
        chars = sorted(list(set(content)))
       
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, scaffold = self.data[idx],  self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()
        
        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern = r"(\[SOS]|\[EOS]|\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<')*(self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles=regex.findall(smiles)

        scaffold += str('<')*(self.scaf_max_len - len(regex.findall(scaffold)))
        
        if len(regex.findall(scaffold)) > self.scaf_max_len:
            scaffold = scaffold[:self.scaf_max_len]

        scaffold=regex.findall(scaffold)
        
        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) 

        if self.sca is not None:
            return x, y, sca_tensor   
        else:
            return x, y
        
class Graph_SMILESDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        self.csv_file = csv_file
        super(Graph_SMILESDataset, self).__init__(root, transform, pre_transform)
        
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
    @property
    def raw_file_names(self):
        return [self.csv_file] 

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        csv_path = os.path.join(self.root, self.csv_file)
        df = pd.read_csv(csv_path)
        error_count = 0

        for i, row in df.iterrows():
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                error_count += 1
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)

            try:
                AllChem.UFFOptimizeMolecule(mol)
            except ValueError:
                print(f"Conformer error for molecule: {smiles}")
                error_count += 1
                continue

            data = from_smiles(smiles)
            data.y = torch.tensor([[row['affinity']]], dtype=torch.float) 

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Number of molecules with errors: {error_count}")
        
    def featurizer(smiles_list):
        data_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  

            data = from_smiles(smiles)

            data_list.append(data)
        return data_list