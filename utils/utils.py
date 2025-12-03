import re
import json
import random
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from moses.utils import get_mol
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, scaffold = None):

    block_size = model.get_block_size()
    model.eval()
    
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] 
        
        logits, _, _ = model(x_cond,  scaffold = scaffold)  

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        
        probs = F.softmax(logits, dim=-1)
                
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, ix), dim=1)
    return x

def check_novelty(gen_smiles, train_smiles): 
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  
        novel = len(gen_smiles) - sum(duplicates)  
        novel_ratio = novel*100./len(gen_smiles) 
    print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

class Iterator(object):
    """Abstract base class for data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class SmilesIterator(Iterator):
    """Iterator yielding data from a SMILES array.
    # Arguments
        x: Numpy array of SMILES input data.
        y: Numpy array of targets data.
        smiles_data_generator: Instance of `SmilesEnumerator`
            to use for random SMILES generation.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        dtype: dtype to use for returned batch. Set to keras.backend.floatx if using Keras
    """

    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + [ self.smiles_data_generator.pad, self.smiles_data_generator._charlen]), dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j+1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

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

      
    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """       
        smiles = []
        for v in vect:
            #mask v 
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)
    
    
def load_config_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        config_data = json.load(json_file)
    return config_data

def smiles_to_canonicsmiles(smiles_list):
    valid=[]
    invalid=[]
    canon_list=[]
    for i in smiles_list: 
        mol = get_mol(i)
        if mol is None:
            invalid.append(mol)
        else:
            valid.append(mol)
    for i in valid:
        smiles=Chem.MolToSmiles(i)
        canon_list.append(smiles)
    return canon_list

def get_valid_canons(smilist):

    canons = []
    invalid_ids = []
    for i, smi in enumerate(smilist):
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            invalid_ids.append(i)
            canons.append(None)
        else:
            canons.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    re_canons = []
    for i, smi in enumerate(canons):
        if smi == None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            print("rdkit bug occurred!!")
            invalid_ids.append(i)
        else:
            re_canons.append(smi)
    return re_canons, invalid_ids


def get_valid_smiles(smilist):
    valid_smiles = []
    invalid_ids = []
    
    for i, smi in enumerate(smilist):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_ids.append(i)
        else:
            valid_smiles.append(smi)  
    
    return valid_smiles, invalid_ids

def extract_scaffold(smiles_list):
    scaffolds = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds.append(scaffold_smiles)
        else:
            scaffolds.append(None)  
    return scaffolds

def filter_large_smiles(smiles_list,limit):
    filtered_smiles = []
    pattern = r"(\[SOS]|\[EOS]|\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    for smiles in smiles_list:
        tokens = regex.findall(smiles)
        if len(tokens) < limit:
            filtered_smiles.append(smiles)
    return filtered_smiles

def filter_no_vocab(smiles_list,stoi):
    valid_smiles = []
    pattern = r"(\[SOS]|\[EOS]|\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    for smiles in smiles_list:
        tokens = regex.findall(smiles)
        if all(token in stoi for token in tokens):
            valid_smiles.append(smiles)
            
    return valid_smiles

def get_best_epoch(file_path, variable_name):
    best_epoch = None
    with open(file_path, 'r') as f:
        for line in f:
            pattern = rf'{variable_name}\s*=\s*(\d+)'
            match = re.search(pattern, line)
            if match:
                best_epoch = int(match.group(1))
                break
    return best_epoch

def save_checkpoint(model, model_save_path):
    logger = logging.getLogger(__name__)
    raw_model = model.module if hasattr(model, "module") else model
    logger.info(f"saving {model_save_path}")
    torch.save(raw_model.state_dict(), model_save_path)

def update_result_value(file_path, key, new_value):
    lines = []
    found = False

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    for i, line in enumerate(lines):
        if line.startswith(key + " ="):
            lines[i] = f"{key} = {new_value}\n"
            found = True
            break

    if not found:
        lines.append(f"{key} = {new_value}\n")
    with open(file_path, 'w') as f:
        f.writelines(lines)
