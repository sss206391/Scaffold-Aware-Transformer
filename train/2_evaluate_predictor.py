"""
Evaluation script for pre-trained Scaffold-Aware Transformer Predictor

This script evaluates the pre-trained predictor model by:
1. Loading the trained model checkpoint
2. Computing test set metrics (loss, MSE, R2)
3. Generating evaluation visualizations

Arguments:
    --gpu: GPU device number (default: 0)
    --project_name: Name of project (default: 'predictor_pik3ca')
                   Use 'predictor_kor' for KOR dataset
                   Use 'predictor_pik3ca' for PIK3CA dataset
                   
Usage:
    python train/2_evaluate_predictor.py --gpu 0 --project_name predictor_pik3ca
    python train/2_evaluate_predictor.py --gpu 0 --project_name predictor_kor
"""
import argparse
import os
import re
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_max_pool as gmp
from utils.dataset import Graph_SMILESDataset
from utils.predictor_model import TrainConfig, GATConfig, GATNet, train, test
from utils.utils import set_seed
from utils.evaluation import predictor_evaluate
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate pre-trained Scaffold-Aware Transformer Predictor'
    )

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    # Project
    parser.add_argument('--project_name', type=str, default='predictor_pik3ca',
                       help='Name of project')
    
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
        bioassay_data_name = "KOR"
    else:
        model_save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor")
        dataset_path_folder = os.path.join(project_root, "dataset/pik3ca")
        dataset_name = "pik3ca_affinity.csv"
        dataset_path = os.path.join(project_root, "dataset/pik3ca/pik3ca_affinity.csv")
        params_path = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/pik3ca_best_params.json")
        bioassay_data_name = "PIK3CA"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  
    
    seed = 42
    set_seed(seed)

    # Load dataset
    data = pd.read_csv(dataset_path)
    dataset = Graph_SMILESDataset(root=dataset_path_folder , csv_file=dataset_name)
    dataset_size = len(dataset)

    # Split dataset
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator)
    
    # Create DataLoaders
    batch_size=32
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    # Load model configuration
    json_path = os.path.join(model_save_path, f"{args.project_name}.json")
    with open(json_path, 'r') as json_file:
        config_dict = json.load(json_file)

    # Create model
    Model_config = GATConfig(
        num_features=config_dict['num_features'],
        n_filters= config_dict['n_filters'],
        embed_dim=config_dict['embed_dim'],
        output_dim=config_dict['output_dim'],
        dropout=config_dict['dropout'],
        num_heads=config_dict['num_heads'],
        concat=config_dict['concat']
    )
    model = GATNet(Model_config).to(device)
    
    # Evaluate on test set
    criterion = torch.nn.MSELoss()
    
    checkpoint = torch.load(os.path.join(model_save_path, f'{args.project_name}.pt'), map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    test_loss, mse, r2 = test(model,test_loader, device, criterion)
    print(f'Final test Loss: {test_loss}, MSE: {mse}, R2 Score: {r2}')
    
    predictor_evaluate(model, test_loader, device, model_save_path, bioassay_data_name)