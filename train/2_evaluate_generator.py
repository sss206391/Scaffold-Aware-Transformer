"""
Evaluation script for pre-trained Scaffold-Aware Transformer Generator

This script evaluates the pre-trained generator by:
1. Loading the trained model checkpoint
2. Generating molecules conditioned on specified scaffolds
3. Computing validity, uniqueness, and novelty metrics

Arguments:
    --gpu: GPU device number (default: 0)
    --batch_size: Batch size for generation (default: 200)
    --gen_size: Total number of molecules to generate per scaffold (default: 10000)
    
Usage:
    python train/2_evaluate_generator.py --gpu 0 --batch_size 200 --gen_size 10000
"""
import argparse
import os
import re
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from utils.generator_model import Transformer, TransformerConfig
from utils.utils import check_novelty, sample, canonic_smiles, set_seed, top_k_logits
from utils.trainer import TrainConfig, run_epoch
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit import RDLogger
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate pre-trained Scaffold-Aware Transformer Generator'
    )

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch size for molecule generation')
    parser.add_argument('--gen_size', type=int, default=10000,
                       help='Total number of molecules to generate per scaffold')
    
    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    SEED=42
    set_seed(SEED)

    # Configure paths
    project_name = 'pre_generator'
    model_save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator")
    sample_path = model_save_path
    vocab_path = os.path.join(project_root, "utils/guacamol_stoi.json")
    dataset_path = os.path.join(project_root, "dataset/guacamol2.csv")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  
    
    data = pd.read_csv(dataset_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    # Load vocabulary
    stoi = json.load(open(vocab_path, 'r'))
    itos = {i: ch for ch, i in stoi.items()}

    # Load model configuration
    json_path = os.path.join(model_save_path, f"{project_name}.json")
    with open(json_path, 'r') as json_file:
        config_dict = json.load(json_file)

    # Prepare data
    smiles = data[data['source']!='test']['smiles']
    scaf = data[data['source']!='test']['scaffold_smiles']
    
    context = '[SOS]'

    # SMILES tokenization pattern 
    pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    Train_config = TrainConfig(save_ckpt=os.path.join(model_save_path, f'{project_name}.pt'))

    # Create model
    Model_config = TransformerConfig(
        vocab_size=config_dict['vocab_size'],
        block_size=config_dict['block_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_embd=config_dict['n_embd'],
        embd_pdrop=config_dict['embd_pdrop'],
        resid_pdrop=config_dict['resid_pdrop'], 
        attn_pdrop=config_dict['attn_pdrop'],
        weight_decay=config_dict['weight_decay'],
        betas=tuple(config_dict['betas']),
        scaffold=config_dict['scaffold'],
        lstm=config_dict['lstm'],     
        scaffold_maxlen=config_dict['scaffold_maxlen']
    )
    model = Transformer(Model_config)

    # Load checkpoint
    checkpoint = torch.load(Train_config.save_ckpt)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    gen_iter = math.ceil(args.gen_size / args.batch_size)

    # Define scaffold conditions for evaluation
    scaf_condition = ['O=C(Cc1ccccc1)NCc1ccccc1', 'c1cnc2[nH]ccc2c1', 'c1ccc(-c2ccnnc2)cc1', 'c1ccc(-n2cnc3ccccc32)cc1', 'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1']
    scaf_condition = [ i + str('<')*(config_dict['scaffold_maxlen'] - len(regex.findall(i))) for i in scaf_condition]
    
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Generation loop
    all_dfs = []
    all_metrics = []
    count=0
    
    for j in scaf_condition:
        molecules = []
        count += 1
        for i in tqdm(range(gen_iter)):
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)
            sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)        
            y = sample(model, x, steps=config_dict['block_size'], temperature=0.9, sample=True, top_k=None, scaffold = sca)
    
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '')
                completion = completion.replace('[SOS]', '')
                completion = completion.replace('[EOS]', '')
                mol = Chem.MolFromSmiles(completion)
                if mol:
                    molecules.append(mol)
                    
        mol_dict = [{'molecule': mol, 'smiles': Chem.MolToSmiles(mol)} for mol in molecules]
        results = pd.DataFrame(mol_dict)
    
        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        
        novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))
    
        print(f'Scaffold: {j}')
        print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
        print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio / 100, 3))
    
        results['scaffold_cond'] = j
        results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
        results['unique'] = np.round(len(unique_smiles) / len(results), 3)
        results['novelty'] = np.round(novel_ratio / 100, 3)
        all_dfs.append(results)
        
        del molecules, mol_dict, results, canon_smiles, unique_smiles, novel_ratio 
        torch.cuda.empty_cache() 
    
    smiles_results = pd.concat(all_dfs)
    smiles_results.to_csv(os.path.join(sample_path, f'{project_name}_scaffold.csv'),index=False)
    
    unique_smiles = list(set(smiles_results['smiles']))
    canon_smiles = [canonic_smiles(s) for s in smiles_results['smiles']]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))
    print('Results------------------------------------------------------------------------------------------')
    print('Valid ratio: ', np.round(len(smiles_results)/(args.batch_size*gen_iter*count), 3))
    print('Unique ratio: ', np.round(len(unique_smiles)/len(smiles_results), 3))
    print('Novelty ratio: ', np.round(novel_ratio/100, 3))