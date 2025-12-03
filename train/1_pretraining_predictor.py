"""
Pre-training script for Scaffold-Aware Transformer Predictor

This script trains the GAT-based predictor model for binding affnity prediction.

Arguments:
    --gpu: GPU device number (default: 0)
    --project_name: Name of project (default: 'predictor_pik3ca')
                   Use 'predictor_kor' for KOR dataset
                   Use 'predictor_pik3ca' for PIK3CA dataset
    --n_filters: Number of initial filters in GAT (default: 53 for PIK3CA, 27 for KOR)
    --embed_dim: Embedding dimension (default: 105 for PIK3CA, 176 for KOR)
    --output_dim: Output dimension (default: 165 for PIK3CA, 70 for KOR)
    --num_heads: Number of attention heads (default: 10)
    --max_epochs: Number of training epochs (default: 100)
    --batch_size: Training batch size (default: 32)
    --lr: Learning rate (default: 3e-4)
    
Usage:
    python train/1_pretraining_predictor.py --gpu 0 --project_name predictor_pik3ca --max_epochs 100 --batch_size 32
    python train/1_pretraining_predictor.py --gpu 0 --project_name predictor_kor --max_epochs 100 --batch_size 32
"""

import argparse
import os
import re
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.cuda.amp import GradScaler
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch_geometric.nn import GATConv, GCNConv, global_max_pool as gmp
from utils.dataset import Graph_SMILESDataset
from utils.predictor_model import TrainConfig, GATConfig, GATNet, train, test
from utils.utils import set_seed, save_checkpoint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-train Scaffold-Aware Transformer Predictor'
    )
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    # Project
    parser.add_argument('--project_name', type=str, default='predictor_pik3ca',
                       help='Name of project')
    # Model 
    parser.add_argument('--n_filters', type=int, default=53,
                       help='Number of initial filters in GAT (default: 53 for PIK3CA, 27 for KOR)')
    parser.add_argument('--embed_dim', type=int, default=105,
                       help='Embedding dimension (default: 105 for PIK3CA, 176 for KOR)')
    parser.add_argument('--output_dim', type=int, default=165,
                       help='Output dimension (default: 165 for PIK3CA, 70 for KOR)')
    parser.add_argument('--num_heads', type=int, default=10,
                       help='Number of attention heads')

    # Training
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Configure paths based on project name
    if "kor" in args.project_name:
        model_save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor")
        dataset_path_folder = os.path.join(project_root, "dataset/kor")
        dataset_name = "kor_affinity.csv"
        dataset_path = os.path.join(project_root, "dataset/kor/kor_affinity.csv")
        params_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/kor_best_params.json")
    else:
        model_save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor")
        dataset_path_folder = os.path.join(project_root, "dataset/pik3ca")
        dataset_name = "pik3ca_affinity.csv"
        dataset_path = os.path.join(project_root, "dataset/pik3ca/pik3ca_affinity.csv")
        params_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/pik3ca_best_params.json")
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)   
    
    seed = 42
    set_seed(seed)

    # Load dataset
    data = pd.read_csv(dataset_path)
    dataset = Graph_SMILESDataset(root=dataset_path_folder, csv_file=dataset_name)
    dataset_size = len(dataset)

    # Split dataset
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator)

    # Get number of node features
    num_features = dataset.num_node_features

    # Set model hyperparameters based on project name or use provided values
    if "kor" in args.project_name:
        n_filters = args.n_filters if args.n_filters is not None else 27
        embed_dim = args.embed_dim if args.embed_dim is not None else 176
        output_dim = args.output_dim if args.output_dim is not None else 70
        dropout = 0.2
        num_heads = args.num_heads 
    else:
        n_filters = args.n_filters if args.n_filters is not None else 53
        embed_dim = args.embed_dim if args.embed_dim is not None else 105
        output_dim = args.output_dim if args.output_dim is not None else 165
        dropout = 0.3
        num_heads = args.num_heads 
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    
    criterion = torch.nn.MSELoss()

    # Save dataset indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    json_path = os.path.join(model_save_path, f"{args.project_name}_data.json")
    
    indices = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    with open(json_path, 'w') as f:
        json.dump(indices, f)

    # Create model configuration
    Model_config = GATConfig(
        num_features=num_features,
        n_filters=args.n_filters,
        embed_dim=args.embed_dim,
        output_dim=args.output_dim,
        dropout=dropout,
        num_heads=args.num_heads,
        concat=False
    )
    
    model = GATNet(Model_config).to(device)

    # Save model configuration
    config_dict = {k: v for k, v in Model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    config_dict['seed'] = seed
    
    json_path = os.path.join(model_save_path, f"{args.project_name}.json")
    with open(json_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

    # Training configuration
    Train_config = TrainConfig(max_epochs=args.max_epochs, 
                          batch_size=args.batch_size, 
                          learning_rate=args.lr,
                          lr_decay=True, 
                          num_workers=10, 
                          save_ckpt=os.path.join(model_save_path, f'{args.project_name}.pt'))

    # Setup optimizer and scaler
    model.to(device)
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizers(Train_config)
    scaler = GradScaler()

    # Training loop
    best_loss = float('inf')  
    train_loss_list=[]
    test_loss_list=[]
    
    best_r2= -1
    best_mse=float('inf')
    
    for epoch in range(args.max_epochs):
        train_loss = train('train',model,train_loader,test_loader,optimizer,scaler, device,epoch,Train_config, criterion)
        train_loss_list.append(train_loss)
    
        test_loss, mse, r2 = train('test',model,train_loader,test_loader,optimizer,scaler, device,epoch,Train_config, criterion)
        test_loss_list.append(test_loss)
        
        good_model = val_loader is None or test_loss < best_loss
        
        if Train_config.save_ckpt is not None and good_model:
            best_loss = test_loss
            best_mse=mse
            best_r2=r2
            save_checkpoint(model, Train_config.save_ckpt)