"""
Fine-tuning script for Scaffold-Aware Transformer Framework

Arguments:
    --gpu: GPU device number (default: 0)
    --project_name: Name of project (kor or pik3ca) (default: pik3ca)
    --memory_size: Size of experience memory (default: 50000)
    --gen_size: Number of molecules to generate per iteration (default: 20000)
    --batch_size: Training batch size (default: 200)
    --max_epochs: Number of fine-tuning epochs (default: 10)
    --save_period: Save checkpoint every N epochs (default: 2)
Usage:
    python train/3_fine_tuning.py --gpu 0 --project_name kor
    python train/3_fine_tuning.py --gpu 0 --project_name pik3ca
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
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from utils.dataset import SmileDataset, Graph_SMILESDataset
from utils.generator_model2 import Transformer, TransformerConfig
from utils.predictor_model import GATConfig, GATNet
from utils import utils, trainer, chemistry, fine_tuning
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.utils import from_smiles
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader as DL
from rdkit import Chem
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune Scaffold-Aware Transformer with predictor guidance'
    )
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number')

    # Project
    parser.add_argument('--project_name', type=str, default='pik3ca',
                       help='Name of project')
    
    # Training
    parser.add_argument('--memory_size', type=int, default=50000,
                       help='Size of experience memory')
    parser.add_argument('--gen_size', type=int, default=20000,
                       help='Number of molecules to generate per iteration')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Training batch size')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--save_period', type=int, default=2,
                       help='Save model checkpoint')
    
    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    SEED=42
    utils.set_seed(SEED)
    
    save_size=args.gen_size
    exp_size=save_size

    # Configure paths based on project name
    if "kor" in args.project_name:
        pretrained_p = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor.pt")
        pretrained_p_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor.json")
        pretrained_p_d = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_kor_data.json")
        save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/fine_tuning/kor")
    else:
        pretrained_p = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca.pt")
        pretrained_p_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca.json")
        pretrained_p_d = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_predictor/predictor_pik3ca_data.json")
        save_path = os.path.join(project_root, "Scaffold_Aware_Transformer/fine_tuning/pik3ca")
    
    vocab_path = os.path.join(project_root, "utils/guacamol_stoi.json")
    sample_smiles = os.path.join(project_root, "Scaffold_Aware_Transformer/pre_generator_scaffold.csv")
    pretrained_g = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator/pre_generator.pt")
    pretrained_g_h = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator/pre_generator.json")
    pretrained_g_t = os.path.join(project_root, "Scaffold_Aware_Transformer/pretrain_generator/Train_config.json")
    dataset_path = os.path.join(project_root, "dataset/guacamol2.csv")
    dataset_raw_path = os.path.join(project_root, "dataset/guacamol_raw.smi")
    
    max_len=102
    scaffold_max_len=102
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Setup featurizer and tokenizer
    featurizer = Graph_SMILESDataset.featurizer
    pattern = r"(\[SOS]|\[EOS]|\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    with open(vocab_path, 'r') as f:
        json_data = json.load(f)
    vocab = list(json_data.keys())
    
    stoi = json.load(open(vocab_path, 'r'))
    itos = {i: ch for ch, i in stoi.items()}

    # Load generator configuration and model
    with open(pretrained_g_h, 'r') as json_file:
        generator_config_dict = json.load(json_file)

    generator_model_config = TransformerConfig(
        vocab_size=generator_config_dict['vocab_size'],
        block_size=generator_config_dict['block_size'],
        n_layer=generator_config_dict['n_layer'],
        n_head=generator_config_dict['n_head'],
        n_embd=generator_config_dict['n_embd'],
        embd_pdrop=generator_config_dict['embd_pdrop'],
        resid_pdrop=generator_config_dict['resid_pdrop'], 
        attn_pdrop=generator_config_dict['attn_pdrop'],
        weight_decay=generator_config_dict['weight_decay'],
        betas=tuple(generator_config_dict['betas']),
        scaffold=generator_config_dict['scaffold'],
        lstm=generator_config_dict['lstm'],     
        scaffold_maxlen=generator_config_dict['scaffold_maxlen']
    )
    
    generator = Transformer(generator_model_config)
    
    checkpoint = torch.load(pretrained_g,map_location=device)
    generator.load_state_dict(checkpoint)
    generator.to(device)

    # Setup generator training config
    generator_tconfig=utils.load_config_from_json(pretrained_g_t)
    generator_tconfig = trainer.TrainConfig(
        max_epochs=generator_tconfig['max_epochs'],
        batch_size=generator_tconfig['batch_size'],
        learning_rate=1e-4,
        lr_decay=generator_tconfig['lr_decay'],
        warmup_tokens=generator_tconfig['warmup_tokens'],
        final_tokens=generator_tconfig['final_tokens'],
        num_workers=generator_tconfig['num_workers'],
        save_ckpt=generator_tconfig['save_ckpt'],
        block_size=generator_tconfig['block_size']
    )
    
    agent_optimizer = generator.configure_optimizers(generator_tconfig)

    # Load predictor configuration and model
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

    # Setup prior generator (frozen)
    prior_generator = Transformer(generator_model_config)
    checkpoint = torch.load(pretrained_g,map_location=device)
    prior_generator.load_state_dict(checkpoint)
    prior_generator.to(device)
    
    for param in prior_generator.parameters():
        param.requires_grad = False

    # Setup scaffold conditions
    scaffold_max_len=generator_config_dict['scaffold_maxlen']
    scaf_condition=['c1cnc2c(N3CCCNCC3)cccc2c1',
    'c1ccc(C2=NOCC2)cc1','c1ccc(Cc2cc3c(CNC4CCCCC4)cccc3o2)cc1',
    'c1ccc(CCCNC2CCCCC2OCCCc2ccccc2)cc1',
    'O=C(CCN(Cc1ccccc1)c1cccc(-c2ccccc2)c1)N1CCN(c2ccccc2)CC1'
    ]
    scaf_condition = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf_condition]
    
    # Initialize experience memory
    step=int(args.memory_size/args.batch_size)+1
    samples = fine_tuning.sample_initial_memory(generator, step,scaf_condition,args.batch_size,stoi,generator_model_config.block_size-2,device)
    
    prior_savings=samples[:save_size]
    prior_sample_filename = f"{args.project_name}_samples_prior.txt"
    prior_sample_save_path = os.path.join(save_path, prior_sample_filename)
    with open(prior_sample_save_path, 'w') as f:
        f.writelines([line + '\n' for line in prior_savings])
        print("prior_sample_save_path : ",prior_sample_save_path)  
    
    _vacans, _ = chemistry.get_valid_canons(samples)
    _vacans=utils.filter_large_smiles(_vacans,generator_model_config.block_size-2)
    _vacans=utils.filter_no_vocab(_vacans,stoi)
    mem_init = list(set(_vacans))[:args.memory_size]

    # Predict activity for initial memory
    mem_data_list = featurizer(mem_init)
    mem_loader = DL(mem_data_list, batch_size=args.batch_size)
    mem_preds = []
    with torch.no_grad():
        for data in mem_loader:
            data = data.to(device)
            output, _, _, _ = predictor(data)
            mem_preds.extend(output.cpu().numpy())
            del data,output
    torch.cuda.empty_cache()        
    mem_preds = np.array(mem_preds).squeeze()

    # Get NLLs from prior
    mem_prior_nlls=trainer.get_NLLs_batch_transformer(mem_init,
                                              prior_generator,
                                              vocab,stoi,
                                              regex,max_len,scaffold_max_len,
                                              args.batch_size,device)

    # Initialize experience memory
    init_mem_dict = { 'smiles': mem_init, 'activity': mem_preds }
    init_mem_dict['prior_nll'] = mem_prior_nlls
    expmem = fine_tuning.ExperienceMemory(init_mem_dict, priority_column='activity')

    # Fine-tuning loop
    for epo in range(0,args.max_epochs+1):
        print("----- epoch: ", epo)
        list_index=-1
        decoded_samples = []
        gen_iter=int(args.memory_size/args.batch_size)
        generator.eval()
        
        for i in tqdm(range(gen_iter)):
            context='[SOS]'
            list_index=list_index+1
            if list_index==5:
                list_index=0
    
            sca_str = scaf_condition[list_index]
            
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)
            sca = torch.tensor([stoi[s] for s in regex.findall(sca_str)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)   
        
            y = utils.sample(generator, x,generator_model_config.block_size-2, temperature=0.9, sample=True,top_k=50, scaffold = sca)
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '').replace('[SOS]', '').replace('[EOS]', '')
                decoded_samples.append(completion)
            
            del x,sca,y,completion
        torch.cuda.empty_cache()
            
        savings = decoded_samples[:save_size]

        # Save checkpoint periodically
        if epo % args.save_period == 0:
            model_filename = f"{args.project_name}_{epo}.pt"
            sample_filename = f"{args.project_name}_samples_{epo}.txt"
            memory_filename = f"{args.project_name}_memory_{epo}.csv"
    
            model_save_path = os.path.join(save_path, model_filename)
            sample_save_path = os.path.join(save_path, sample_filename)
            memory_save_path = os.path.join(save_path, memory_filename)
    
            ckpt_dict = {
                'config': generator_model_config.__dict__,
                'epoch': epo,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': agent_optimizer.state_dict()
            }
    
            torch.save(ckpt_dict, model_save_path)
    
            with open(sample_save_path, 'w') as f:
                f.writelines([line + '\n' for line in savings])
    
            expmem.memory.to_csv(memory_save_path, index=False)

        # Filter valid molecules
        vacans, _ = utils.get_valid_canons(decoded_samples)
        vacans=utils.filter_large_smiles(vacans,generator_model_config.block_size-2)
        vacans=utils.filter_no_vocab(vacans,stoi)
        vaunis = list(set(vacans))

        # Predict activity
        vauni_data_list = featurizer(vaunis)
        vauni_loader = DL(vauni_data_list, batch_size=args.batch_size)
        vauni_preds = []
        with torch.no_grad():
            for data in vauni_loader:
                data = data.to(device)
                output, _, _, _ = predictor(data)
                vauni_preds.extend(output.cpu().numpy())
                del data, output
        torch.cuda.empty_cache()
    
        vauni_preds = np.array(vauni_preds).squeeze()
    
        print("-> num uniq: ", len(vaunis)) # debugging
        print("-> avg pred_act: ", vauni_preds.mean())

                
        vauni_prior_nlls= trainer.get_NLLs_batch_transformer(vaunis,
                                                     prior_generator, 
                                                     vocab, 
                                                     stoi,regex,max_len,scaffold_max_len,args.batch_size,device)

        mem_inds, mem_samp = expmem.sample(exp_size)
        mem_smis, mem_preds  = mem_samp['smiles'], mem_samp['activity']
        mem_prior_nlls = mem_samp['prior_nll']

        cmptrs = np.concatenate((vaunis, mem_smis))
        cmptrs_size = len(cmptrs)
        cmptrs_preds = np.concatenate((vauni_preds, mem_preds))
        
        cmptrs_prior_nlls = np.concatenate((vauni_prior_nlls, mem_prior_nlls))
        
        cmptrs_agent_nlls = trainer.get_NLLs_batch_transformer(cmptrs, 
                                                        generator,
                                                        vocab,
                                                        stoi,
                                                        regex,
                                                        max_len,
                                                        scaffold_max_len,
                                                        args.batch_size,
                                                        device)

        # Tournament selection and memory update
        list_scores = np.array([cmptrs_preds, cmptrs_agent_nlls, -cmptrs_prior_nlls])
        surv_sizes = [int(cmptrs_size/2), int(cmptrs_size/4), int(cmptrs_size/8)]
        
        laytour = fine_tuning.LayeredTournaments(list_scores, surv_sizes)
        survivors = laytour.perform_tournaments() # indices of the final winners
    
        winners_dict = { 'smiles':cmptrs[survivors], 'activity':cmptrs_preds[survivors] }
        winners_dict['prior_nll'] = cmptrs_prior_nlls[survivors]
        expmem.update(winners_dict)
        
        new_tsmiles=list(cmptrs[survivors])
        
        new_tsmiles=utils.filter_large_smiles(new_tsmiles,generator_model_config.block_size-2)
        new_tscaffold=utils.extract_scaffold(new_tsmiles)
    
        lens = [len(regex.findall(i.strip())) for i in (new_tsmiles)]
        new_max_len = max(lens)+2
    
        lens = [len(regex.findall(i.strip())) for i in (new_tscaffold)]
        new_scaffold_max_len = max(lens)+2
    
        new_tsmiles = ['[SOS]' + i + '[EOS]' for i in new_tsmiles]
        new_tscaffold = ['[SOS]' + i + '[EOS]' for i in new_tscaffold]
    
        new_tsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in new_tsmiles]
    
        new_tscaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in new_tscaffold]

        # Train generator
        train_dataset = SmileDataset(False,new_tsmiles, vocab, 
                                     max_len, aug_prob=0, scaffold=new_tscaffold, 
                                     scaffold_maxlen=scaffold_max_len)
    
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=10)
        scaler = GradScaler()
        tokens=0
        train_loss,tokens= trainer.fine_tuning_run(
            'train', generator, train_dataset,agent_optimizer, scaler, device, epo, generator_tconfig, tokens, stoi, itos
        )
        del train_dataset, train_loader, scaler
        torch.cuda.empty_cache()