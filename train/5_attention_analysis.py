"""
Attention Analysis Script

This script performs attention-based analysis on molecules to identify important substructures that contribute to predicted binding affinity.
The analysis uses Graph Attention Network (GAT) attention coefficients to highlight key molecular motifs.

Arguments:
    --gpu: GPU device number (default: 0)
    --project_name: Name of project (kor or pik3ca) (default: 'pik3ca')
    --smiles: SMILES string of molecule to analyze (required)
    --select_mode: Mode for selecting important substructures (default: 'hybrid')
                  Options: 'hybrid', 'edges', 'nodes'
    --top_percent_edges: Percentage of top edges to highlight (default: 0.2)
    --top_percent_nodes: Percentage of top nodes to highlight (default: 0.2)
    --node_weight_direct: Direct weight for node importance (default: 0.0)
    --node_weight_from_edge: Weight from edge importance (default: 0.0)
    --node_reduce_from_edges: Method to reduce edge weights to node (default: 'max')
                             Options: 'max', 'mean', 'sum'
    --require_node_attn: Require node attention (default: True)
    --expand_hops: Number of hops to expand from important nodes (default: 1)
Usage:
    python train/5_attention_analysis.py --gpu 0 --project_name kor --smiles "OC(C1=C2[C@@]34[C@H]5O1)=CC=C2C[C@@H](N(CC4)CC6CC6)[C@]3(O)CCC5=C"
    python train/5_attention_analysis.py --gpu 0 --project_name pik3ca --smiles "COC1=C(C=CC2=C3NCCN3C(=NC(=O)C4=CN=C(N=C4)N)N=C21)OCCCN5CCOCC5"
"""
import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from utils.predictor_model import GATConfig, GATNet
from utils.dataset import Graph_SMILESDataset
from utils.utils import set_seed
from utils import chemistry, attention_analysis
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.utils import from_smiles
from torch_geometric.nn import GATConv
from tqdm.auto import tqdm
from rdkit.Chem import Draw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Attention-based analysis of molecular substructures'
    )
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    
    # Project
    parser.add_argument('--project_name', type=str, default='pik3ca',
                       help='Name of project (kor or pik3ca)')

    # Input
    parser.add_argument('--smiles', type=str, required=True,
                       help='SMILES string of molecule to analyze')

    # Analysis Parameters
    parser.add_argument('--select_mode', type=str, default='hybrid',
                       choices=['hybrid', 'edges', 'nodes'],
                       help='Mode for selecting important substructures')
    
    parser.add_argument('--top_percent_edges', type=float, default=0.2,
                       help='Percentage of top edges to highlight (0.0-1.0)')
    
    parser.add_argument('--top_percent_nodes', type=float, default=0.2,
                       help='Percentage of top nodes to highlight (0.0-1.0)')
    
    parser.add_argument('--node_weight_direct', type=float, default=0.0,
                       help='Direct weight for node importance')
    
    parser.add_argument('--node_weight_from_edge', type=float, default=0.0,
                       help='Weight from edge importance to node')
    
    parser.add_argument('--node_reduce_from_edges', type=str, default='max',
                       choices=['max', 'mean', 'sum'],
                       help='Method to reduce edge weights to node')
    
    parser.add_argument('--require_node_attn', action='store_true', default=True,
                       help='Require node attention')
    
    parser.add_argument('--expand_hops', type=int, default=1,
                       help='Number of hops to expand from important nodes')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Set random seed
    seed = 42
    set_seed(seed)

    # Configure paths based on project name
    if "kor" in args.project_name:
        pretrained_p = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor.pt")
        pretrained_p_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor.json")
    else:
        pretrained_p = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca.pt")
        pretrained_p_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca.json")
    
    save_path = os.path.join(project_root, "attention_analysis")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load predictor configuration
    with open(pretrained_p_h, 'r') as json_file:
        predictor_config_dict = json.load(json_file)
    
    featurizer = Graph_SMILESDataset.featurizer
    predictor_model_config= GATConfig(
        num_features=predictor_config_dict['num_features'],
        n_filters=predictor_config_dict['n_filters'],
        embed_dim=predictor_config_dict['embed_dim'],
        output_dim=predictor_config_dict['output_dim'],
        dropout=predictor_config_dict['dropout'],
        num_heads=predictor_config_dict['num_heads'],
        concat=predictor_config_dict['concat']
    )
    predictor = GATNet(predictor_model_config)
    checkpoint = torch.load(pretrained_p,map_location=device)
    predictor.load_state_dict(checkpoint)
    predictor.to(device)
    predictor.eval()

    smiles_for_filename = args.smiles.replace('/', '_').replace('\\', '_').replace('[', '').replace(']', '')
    if len(smiles_for_filename) > 50:
        smiles_for_filename = smiles_for_filename[:50]  # Truncate if too long
        
    # Perform attention analysis
    out = attention_analysis.Attention_analysis(
        args.smiles,
        predictor,
        featurizer,
        save_dir=save_path,
        prefix=f"{args.project_name}_{smiles_for_filename}",
        select_mode=args.select_mode,
        top_percent_edges=args.top_percent_edges,
        top_percent_nodes=args.top_percent_nodes,
        node_weight_direct=args.node_weight_direct,
        node_weight_from_edge=args.node_weight_from_edge,
        node_reduce_from_edges=args.node_reduce_from_edges,
        require_node_attn=args.require_node_attn,
        expand_hops=args.expand_hops,
        img_width=1000,
        base_height=560,
        font_scale=1.4,
        atom_radius=0.45,
        bond_width=3,
        largest_cc=True
    )