"""
    This file includes functions calculating metrics for gpc model evaluation.
    Please check each experiment notebooks, and check the subsidiary files needed to
        perform the following evaluations. 
        e.g. *.smi for valid generations and *.npy for fingerprints.
"""

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from dataclasses import dataclass
from utils.dataset import Graph_SMILESDataset
from torch_geometric.loader import DataLoader as DL
from . import analysis, chemistry, frechet_chemnet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

@dataclass
class EvalConfig:
    ssize : int  # sample size
    vc_smis : list  # valid & canonical smiles generations
    npfps : np.ndarray  # numpy array format of rdkit fingerprint of generations
    simmat_size : int  # size of generations to be used for similarity matrix calculation
    fc_vecs : np.ndarray  # vectors calculated from FreChet ChemNet module
    data_smis : list  # smiles from data, either validation set or test set
    data_rdkfps : list  # data rdkit fingerprints
    data_fc_vecs : np.ndarray  # data vectors
    ot_repeats : int  # how many times OT calculation repeats
    
def eval_standard(evcon:EvalConfig, pret_smis):
    """
        Standard metrics for evaluation
    """
    unis = list(set(evcon.vc_smis))  # unique generations
    pret_set = set(pret_smis)
    novs = list(set(unis).difference(pret_set))
    
    validity = len(evcon.vc_smis) / evcon.ssize
    uniqueness = len(unis) / len(evcon.vc_smis)
    novelty = len(novs) / len(unis)
    
    rdkfps = chemistry.np2rdkfps(evcon.npfps[:evcon.simmat_size])
    intdiv = analysis.internal_diversity(rdkfps)
    return validity, uniqueness, novelty, intdiv

def eval_optimization(evcon:EvalConfig, predictor,device):
    """
        Optimization metrics for evaluation
    """
    batch_size=200
    featurizer = Graph_SMILESDataset.featurizer
    predict_data=featurizer(evcon.vc_smis)
    
    pred_loader = DL(predict_data, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for data in pred_loader:
            data = data.to(device)
            output, _, _, _ = predictor(data)
            preds.extend(output.cpu().numpy())
            del data, output
    
    predact = np.mean(preds)
    
    gen_rdkfps = chemistry.np2rdkfps(evcon.npfps)
    ext_simmat = analysis.calculate_simmat(gen_rdkfps[:evcon.simmat_size], evcon.data_rdkfps)
    pwsim = np.mean(ext_simmat)

    fcdval = frechet_chemnet.fcd_calculation(evcon.fc_vecs, evcon.data_fc_vecs)
    
    supply_sz = len(evcon.data_smis) * evcon.ot_repeats
    ot_simmat = analysis.calculate_simmat(gen_rdkfps[:supply_sz], evcon.data_rdkfps)
    ot_distmat = analysis.transport_distmat(analysis.tansim_to_dist, ot_simmat,
                                            num_repeats=evcon.ot_repeats)
    _, _, motds = analysis.repeated_optimal_transport(ot_distmat, evcon.ot_repeats)
    otdval = np.mean(motds)
    return predact, pwsim, fcdval, otdval

def predictor_evaluate(model, test_loader, device, model_save_path, dataset_name):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            output, _, _, _ = model(data)
            all_preds.append(output.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('Actual Values',fontsize=16, fontweight='bold', fontfamily='serif') 
    plt.ylabel('Predicted Values',fontsize=16, fontweight='bold', fontfamily='serif') 
    plt.title(dataset_name,fontsize=18,fontweight='bold',fontfamily='serif')
    plt.grid(True)
    plt.savefig(f"{model_save_path}/{dataset_name}_1.png", format='png')
    
    residuals = all_targets - all_preds
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, edgecolor='k')
    plt.xlabel('Residuals', fontsize=16, fontweight='bold', fontfamily='serif')  
    plt.ylabel('Frequency', fontsize=16, fontweight='bold', fontfamily='serif')  
    plt.title(dataset_name,fontsize=18,fontweight='bold',fontfamily='serif')
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(f"{model_save_path}/{dataset_name}_2.png", format='png')