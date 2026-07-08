"""
Pre-training script for Scaffold-Aware Transformer Generator

This script pre-trains the generator model on the GuacaMol dataset.

Arguments:
    --gpu: GPU device number (default: 0)
    --n_layer: Number of transformer layers (default: 8)
    --n_head: Number of attention heads (default: 8)
    --n_embd: Embedding dimension (default: 512)
    --max_epochs: Number of training epochs (default: 10)
    --batch_size: Training batch size (default: 384)
    --lr: Learning rate (default: 6e-4)
    --num_workers: Number of data loading workers (default: 10)
    
Usage:
    python train/1_pretraining_generator.py --gpu 0 --max_epochs 10 --batch_size 384
"""
import os
import re
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from utils.dataset import SmileDataset
from utils.generator_model import Transformer, TransformerConfig
from utils.utils import set_seed, save_checkpoint
from utils.trainer import TrainConfig,train_config_to_dict,run_epoch
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-train Scaffold-Aware Transformer Generator'
    )

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    # Model
    parser.add_argument('--n_layer', type=int, default=8,
                       help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=512,
                       help='Embedding dimension')

    # Training
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=384,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=10,
                       help='Number of data loading workers')
    
    args = parser.parse_args()

    # Configuration
    project_name = 'pre_generator'
    model_save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator")
    vocab_path = os.path.join(project_root, "utils/guacamol_stoi.json")
    dataset_path = os.path.join(project_root, "dataset/guacamol2.csv")

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Create save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  

    # Load vocabulary
    stoi = json.load(open(vocab_path, 'r'))
    itos = {i: ch for ch, i in stoi.items()}

    with open(vocab_path, 'r') as f:
        json_data = json.load(f)
    
    vocab = list(json_data.keys())
    
    debug=False 
    lstm = False
    SEED =42
    set_seed(SEED)

    # Load and prepare data
    data = pd.read_csv(dataset_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    
    train_data = data[data['source'] == 'train'].reset_index(drop=True)
    val_data = data[data['source'] == 'val'].reset_index(drop=True)

    # Extract SMILES and scaffolds
    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    
    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    # Tokenization pattern
    pattern = r"(\[SOS]|\[EOS]|\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    # Calculate max lengths
    lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)+2
    
    lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)+2

    # Add special tokens
    smiles = ['[SOS]' + i + '[EOS]' for i in smiles]
    vsmiles = ['[SOS]' + i + '[EOS]' for i in vsmiles]
    scaffold = ['[SOS]' + i + '[EOS]' for i in scaffold]
    vscaffold = ['[SOS]' + i + '[EOS]' for i in vscaffold]

    # Padding
    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]
    
    scaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

    # Create datasets
    train_dataset = SmileDataset(debug,smiles, vocab, 
                                 max_len, aug_prob=0, scaffold=scaffold, 
                                 scaffold_maxlen=scaffold_max_len)
    
    valid_dataset = SmileDataset(debug,vsmiles, vocab, max_len,  aug_prob=0, scaffold=vscaffold, scaffold_maxlen= scaffold_max_len)

    # Create model configuration
    Model_config = TransformerConfig(
        vocab_size=len(vocab), 
        block_size=train_dataset.max_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        embd_pdrop=0.1,
        resid_pdrop=0.1, 
        attn_pdrop=0.1,
        weight_decay=0.1,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        scaffold=True,  
        lstm=False,     
        scaffold_maxlen=scaffold_max_len)
    
    model = Transformer(Model_config).to(device)
    
    config_dict = {k: v for k, v in Model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    config_dict['seed'] = SEED
    
    json_path = os.path.join(model_save_path, f"{project_name}.json")
    with open(json_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

    # Training configuration
    Train_config = TrainConfig(max_epochs=args.max_epochs, 
                          batch_size=args.batch_size, 
                          learning_rate=args.lr,
                          lr_decay=True, 
                          warmup_tokens=0.1*len(train_data)*max_len, 
                          final_tokens=args.max_epochs*len(train_data)*max_len,
                          num_workers=args.num_workers, 
                          save_ckpt=os.path.join(model_save_path, f'{project_name}.pt'),
                          block_size=train_dataset.max_len,
                               generate=False
                              )
    
    train_config_dict = train_config_to_dict(Train_config)
    
    json_path = os.path.join(model_save_path, f"Train_config.json")
    with open(json_path, 'w') as json_file:
        json.dump(train_config_dict, json_file, indent=4)
    
    model.to(device)
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizers(Train_config)
    scaler = GradScaler()

    # Training loop
    best_loss = float('inf') 
    tokens = 0
    train_loss_list=[]
    test_loss_list=[]
    
    for epoch in range(args.max_epochs):
        train_loss,tokens= run_epoch('train', model, train_dataset,valid_dataset,optimizer, scaler, device, epoch, Train_config, tokens,  train_dataset.stoi, train_dataset.itos)
        train_loss_list.append(train_loss)
        
        if valid_dataset is not None:
            test_loss, tokens = run_epoch('test', model, train_dataset, valid_dataset,optimizer, scaler, device, epoch, Train_config, tokens, train_dataset.stoi, train_dataset.itos)
            test_loss_list.append(test_loss)
    
        good_model = valid_dataset is None or test_loss < best_loss
        if Train_config.save_ckpt is not None and good_model:
            best_loss = test_loss
            print(f'Saving at epoch {epoch + 1}')
            save_checkpoint(model, Train_config.save_ckpt)
    