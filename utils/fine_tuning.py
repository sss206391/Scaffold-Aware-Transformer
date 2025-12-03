import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm.auto import tqdm
import re
from . import utils
import json
import pandas as pd
def tournament_selection(pool_size, scores, num_winners):
    """
        random tournament selection (returns the indices)
        should be pool_size > num_winners
    """
    winners = []
    cpool = set(range(pool_size))
    for i in range(num_winners):
        competrs = np.random.choice(list(cpool), 2) # two indices are chosen
        if scores[competrs[0]] > scores[competrs[1]]:
            winners.append(competrs[0])
            cpool.remove(competrs[0])
        else:
            winners.append(competrs[1])
            cpool.remove(competrs[1])
    return winners

class LayeredTournaments:
    """
        Fine-tuning set filtering module using mutiple stages of tournaments

        list_scores should be numpy array of (N, m) where N is the number of 
        tournament layers, and m is the number of the competitors.
        survivor_sizes should be a list of integers, specifying the number of 
        survivors at each tournament layer. It should contain N integers.
    """
    def __init__(self, list_scores: np.array, survivor_sizes):
        self.list_scores = list_scores
        self.layers, self.competitors = list_scores.shape
        self.survivor_sizes = survivor_sizes

    def perform_tournaments(self):
        survivors = np.arange(self.competitors)
        for i in range(self.layers):
            scores = (self.list_scores[i])[survivors]
            winners = tournament_selection(len(scores), scores, self.survivor_sizes[i])
            survivors = survivors[winners]
        return survivors

class SelectDeterministic:
    """
        Fine-tuning set filtering module using deterministic selecction.

        list_scores should be numpy array of (N, m) where N is the number of 
        tournament layers, and m is the number of the competitors.
        survivor_sizes should be a list of integers, specifying the number of 
        survivors at each tournament layer. It should contain N integers.
    """
    def __init__(self, list_scores: np.array, survivor_sizes):
        self.list_scores = list_scores
        self.layers, self.competitors = list_scores.shape
        self.survivor_sizes = survivor_sizes

    def perform_tournaments(self):
        survivors = np.arange(self.competitors)
        for i in range(self.layers):
            scores = (self.list_scores[i])[survivors]
            sorted_inds = np.argsort(scores)[::-1]    # sort ascending order -> descending
            winners = sorted_inds[:self.survivor_sizes[i]]   # filter by deterministic selection
            survivors = survivors[winners]
        return survivors

class ExperienceMemory:
    """
        Memorize the previously found 'good' generated examples.
        This module manages a pandas DataFrame. init_memory_dict is a dictionary
        that initializes the memory DF. One notable feature of this memory is that 
        for updates, it removes the ones that have low scores on priority_column.

        'smiles' is required as a column.
        The official columns used in the paper are ['smiles', 'activity', 'prior NLL'].
    """
    def __init__(self, init_memory_dict, priority_column):
        mem_df = pd.DataFrame(init_memory_dict)
        if 'smiles' not in mem_df.columns:
            print(" 'smiles' key/column is required!")
            self.memory = None
            return
        self.priority_column = priority_column
        # sort by priority (ascending order)
        self.memory = mem_df.sort_values(by=priority_column).reset_index(drop=True)

    def sample(self, ssize):
        mem_inds = np.random.choice(len(self.memory), ssize, replace=False)
        return mem_inds, self.memory.iloc[mem_inds].copy()
    
    def update(self, winners_dict):
        """
            delete individuals that have low scores on priority_column,
            and replace them with the new high-scoring individuals (winners).
            ---- Check if the smiles is not already in the memory. ---- we are not doing this
        """
        winners_df = pd.DataFrame(winners_dict)
        self.memory.iloc[:len(winners_df)] = winners_df
        self.memory.sort_values(by=self.priority_column, inplace=True) # ascending order
        self.memory.reset_index(inplace=True, drop=True)
        return

def sample_initial_memory(generator, num_samples,scaf_condition,batch_size,stoi,max_sequence,device):

    pattern = r"(\[SOS]|\[EOS]|\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    itos = {i: ch for ch, i in stoi.items()}
    generated_smiles = []
    context='[SOS]'
    generator.eval()  
    list_index=-1
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            list_index=list_index+1
            if list_index==5:
                list_index=0
            
            sca_str = scaf_condition[list_index]
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(batch_size, 1).to(device)
            sca = torch.tensor([stoi[s] for s in regex.findall(sca_str)], dtype=torch.long)[None,...].repeat(batch_size, 1).to(device)   
            y = utils.sample(generator, x,max_sequence, temperature=0.9, sample=True,top_k=50, scaffold = sca)
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '').replace('[SOS]', '').replace('[EOS]', '')
                generated_smiles.append(completion)
            del y, completion
    del x, sca
    torch.cuda.empty_cache()
    return generated_smiles