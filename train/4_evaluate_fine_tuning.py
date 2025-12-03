"""
Evaluation script for Scaffold-Aware Transformer Framework

This script evaluates the fine-tuned framework by computing various metrics
including validity, uniqueness, novelty, diversity, predicted activity,
pairwise similarity, FCD, and OTD.

Arguments:
    --gpu: GPU device number (default: 0)
    --project_name: Name of project (kor or pik3ca) (default: pik3ca)
    --save_period: Checkpoint saving period used during training (default: 2)    
Usage:
    python train/4_evaluate_fine_tuning.py --gpu 0 --project_name kor
    python train/4_evaluate_fine_tuning.py --gpu 0 --project_name pik3ca
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
import fcd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from utils.dataset import SmileDataset, Graph_SMILESDataset
from utils.generator_model2 import Transformer, TransformerConfig
from utils.predictor_model import GATConfig, GATNet
from utils.trainer import get_NLLs_batch_transformer
from utils import utils, trainer, chemistry, frechet_chemnet, analysis, evaluation
from torch_geometric.data import DataLoader, Data
from torch_geometric.loader import DataLoader as DL
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from typing import List
from tqdm.auto import tqdm
from moses.utils import get_mol

# Force TensorFlow to use CPU (for FCD calculation)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Fine-tuned Scaffold-Aware Transformer'
    )

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')
    
    # Project
    parser.add_argument('--project_name', type=str, default='pik3ca',
                       help='Name of project')
    parser.add_argument('--save_period', type=int, default=2,
                       help='Checkpoint saving period (must match training)')
    
    args = parser.parse_args()

    # Configure paths based on project name
    if "kor" in args.project_name:
        pretrained_p = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor.pt")
        pretrained_p_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor.json")
        pretrained_p_d = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor_data.json")
        save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/fine_tuning/kor")
        bioassay_data = pd.read_csv(os.path.join(project_root, "dataset/kor/kor_affinity.csv"))
    else:
        pretrained_p = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca.pt")
        pretrained_p_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca.json")
        pretrained_p_d = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca_data.json")
        save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/fine_tuning/pik3ca")
        bioassay_data = pd.read_csv(os.path.join(project_root, "dataset/pik3ca/pik3ca_affinity.csv"))
    
    vocab_path = os.path.join(project_root, "utils/guacamol_stoi.json")
    sample_smiles = os.path.join(project_root, "Scaffold_Aware_Transformer/pre_generator_scaffold.csv")
    pretrained_g = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator/pre_generator.pt")
    pretrained_g_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator/pre_generator.json")
    pretrained_g_t = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator/Train_config.json")
    dataset_path = os.path.join(project_root, "dataset/guacamol2.csv")
    dataset_raw_path = os.path.join(project_root, "dataset/guacamol_raw.smi")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Setup file paths
    smiles_path = os.path.join(save_path, f'{args.project_name}_smiles_prior.smi')
    npfps_path = os.path.join(save_path, f'{args.project_name}_npfps_prior.npy')
    fcvec_path = os.path.join(save_path, f'{args.project_name}_fcvec_prior.npy')

    # Load FCD reference model
    fc_ref_model = fcd.load_ref_model()

    # Process prior samples
    with open(os.path.join(save_path, f"{args.project_name}_samples_prior.txt"), 'r') as f: #config.sample_fmt%epo => 
        gens = [line.strip() for line in f.readlines()]
    vcs, invids = utils.get_valid_canons(gens)
    
    print(f"Valid canonical SMILES: {len(vcs)}")
    print(f"Invalid SMILES: {len(invids)}")
    
    with open(smiles_path, 'w') as f:
        f.writelines([line+'\n' for line in vcs])
    fps = chemistry.get_fps_from_smilist(vcs)
    np.save(npfps_path, chemistry.rdk2npfps(fps))
    try:
        fcvecs = fcd.get_predictions(fc_ref_model, vcs)  # ChemNet vectors
        np.save(fcvec_path, fcvecs)
    except IndexError as e:
        print(f"IndexError encountered and skipped at error")

    # Setup paths for epoch-wise evaluation
    smiles_path = os.path.join(save_path, f'{args.project_name}_smiles_e%d.smi')
    npfps_path = os.path.join(save_path, f'{args.project_name}_npfps_e%d.npy')
    fcvec_path = os.path.join(save_path, f'{args.project_name}_fcvec_e%d.npy')

    # Define epochs to evaluate
    epochs = list(range(0,31, args.save_period))

    # Process each epoch
    for epo in epochs:
        print(epo)
        with open(os.path.join(save_path, f"{args.project_name}_samples_{epo}.txt"), 'r') as f: #config.sample_fmt%epo => 
            gens = [line.strip() for line in f.readlines()]
        vcs, invids = utils.get_valid_canons(gens)
        print(f"Invalid SMILES: {len(invids)}")
        
        with open(smiles_path%epo, 'w') as f:
            f.writelines([line+'\n' for line in vcs])
            
        fps = chemistry.get_fps_from_smilist(vcs)
        np.save(npfps_path%epo, chemistry.rdk2npfps(fps))
        try:
            fcvecs = fcd.get_predictions(fc_ref_model, vcs)  # ChemNet vectors
            np.save(fcvec_path % epo, fcvecs)
        except IndexError as e:
            print(f"IndexError encountered and skipped at epoch {epo}: {str(e)}")
            continue
    
    # loading validation dataset
    with open(pretrained_p_d, 'r') as f:
        folds = json.load(f) 
    if "kor" in args.project_name:
        data_npfps = np.load(os.path.join(project_root, "dataset/kor/kor_aff_npfps.npy"))
        data_fcvecs = np.load(os.path.join(project_root, "dataset/kor/kor_aff_fcvec.npy"))
    else:
        data_npfps = np.load(os.path.join(project_root, "dataset/pik3ca/pik3ca_aff_npfps.npy"))
        data_fcvecs = np.load(os.path.join(project_root, "dataset/pik3ca/pik3ca_aff_fcvec.npy"))
    
    val_fold_id = "val"
    val_npfps = data_npfps[folds[val_fold_id]]
    val_rdkfps = chemistry.np2rdkfps(val_npfps)
    val_fcvecs = data_fcvecs[folds[val_fold_id]]

    # Compute FCD and OTD for validation set
    dsize = len(val_rdkfps)  # demand size for OT
    ssize = dsize*10  # supply size for repeated OT   
    
    val_fcd_list = []
    val_otd_list = []
    for epo in epochs:
        try:
            print(epo)
            
            # FCD calculation
            gen_fcvecs = np.load(fcvec_path%epo)
            fcdval = frechet_chemnet.fcd_calculation(val_fcvecs, gen_fcvecs)
            val_fcd_list.append(fcdval)

            # OTD calculation
            gen_npfps = np.load(npfps_path%epo)[:ssize] 
            gen_rdkfps = chemistry.np2rdkfps(gen_npfps)
            simmat = analysis.calculate_simmat(gen_rdkfps, val_rdkfps)  
            distmat = analysis.transport_distmat(analysis.tansim_to_dist, simmat, 10) 
            _, _, motds = analysis.repeated_optimal_transport(distmat, repeat=10)
            val_otd_list.append(np.mean(motds)) 
        except :
            print(f"IndexError encountered and skipped at epoch {epo}:")
            continue

    # Setup evaluation configurations
    SAMPLE_SIZE = 20000  
    INTDIV_SIZE = 1000 
    
    scales = list(range(2, 31, args.save_period))  # [2, 4, 6, 8, ..., 28, 30]
    
    model_names = [f'scale_transformer{scale}' for scale in scales]
    
    perf_table = pd.DataFrame(index=['validity','uniqueness','novelty','diversity','PredAct','PwSim','FCD','OTD'], 
                            columns=model_names)
    metrics = perf_table.index.tolist()

    # Setup file paths
    paths_vc = {}
    paths_npfps = {}
    paths_fc_vecs = {}

    for scale in scales:
        model_name = f'scale_transformer{scale}'
        paths_vc[model_name] = os.path.join(save_path, f"{args.project_name}_smiles_e{scale}.smi")
        paths_npfps[model_name] = os.path.join(save_path, f"{args.project_name}_npfps_e{scale}.npy")
        paths_fc_vecs[model_name] = os.path.join(save_path, f"{args.project_name}_fcvec_e{scale}.npy")

    # Load pre-training dataset
    with open(dataset_raw_path, 'r') as f:
        pret_smis = [line.strip() for line in f.readlines()]
    
    # Load test data
    with open(pretrained_p_d, 'r') as f:
        folds = json.load(f)
    test_ids = folds['test']
    test_data = bioassay_data.iloc[test_ids]
    
    if "kor" in args.project_name:
        tsa_data = test_data[test_data['affinity']>7.0]  # active among test set
    else:
        tsa_data = test_data[test_data['affinity']>8.0]  # active among test set
        
    tsa_smis = tsa_data['smiles'].tolist()
    tsa_rdkfps = chemistry.get_fps_from_smilist(tsa_smis) 
    tsa_fc_vecs = fcd.get_predictions(fc_ref_model, tsa_smis) 
    
    # Create evaluation config objects
    evcons = {}
    for mn in model_names:
        with open(paths_vc[mn], 'r') as f:
            vc_smis = [line.strip() for line in f.readlines()]
            
        npfps = np.load(paths_npfps[mn])
        fc_vecs = np.load(paths_fc_vecs[mn])
        
        evc = evaluation.EvalConfig(
                ssize=SAMPLE_SIZE, vc_smis=vc_smis, npfps=npfps, simmat_size=INTDIV_SIZE, fc_vecs=fc_vecs,
                data_smis=tsa_data, data_rdkfps=tsa_rdkfps, data_fc_vecs=tsa_fc_vecs, ot_repeats=10
        )
        evcons[mn] = evc
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Load predictor
    with open(pretrained_p_h, 'r') as json_file:
        predictor_config_dict = json.load(json_file)
    
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

    # Evaluate and fill out performance table
    for mn in model_names:
        print(mn)
        # Standard metrics
        va, uni, nov, div = evaluation.eval_standard(evcons[mn], pret_smis)

        # Optimization metrics
        predact, pwsim, fcdval, otdval = evaluation.eval_optimization(evcons[mn], predictor, device)

        # Fill performance table
        perf_table[mn]['validity'] = va
        perf_table[mn]['uniqueness'] = uni
        perf_table[mn]['novelty'] = nov
        perf_table[mn]['diversity'] = div
        perf_table[mn]['PredAct'] = predact
        perf_table[mn]['PwSim'] = pwsim
        perf_table[mn]['FCD'] = fcdval
        perf_table[mn]['OTD'] = otdval
        
    # Save performance table
    perf_table.to_csv(os.path.join(save_path, f'{args.project_name}_performance.csv'))
    perf_table