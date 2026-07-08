import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.utils import from_smiles 
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

class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer
    
    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot
