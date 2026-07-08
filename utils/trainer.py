import math
import logging
import numpy as np  
import torch 
import torch.nn.functional as F  
from torch.utils.data import DataLoader 
from tqdm.auto import tqdm
from . import utils
from utils.dataset import SmileDataset
class TrainConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 
    lr_decay = False
    warmup_tokens = 375e6 
    final_tokens = 260e9 
    save_ckpt = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def run_epoch(split, model, train_dataset, test_dataset, 
              optimizer, scaler, device, epoch, 
              config, tokens, stoi, itos):
    is_train = split == 'train'
    model.train(is_train)
    
    data = train_dataset if is_train else test_dataset
    loader = DataLoader(data, shuffle=True, pin_memory=True,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers)
    losses = []
    pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
    
    for it, (x, y, scaffold) in pbar:
        x = x.to(device)
        y = y.to(device)
        scaffold = scaffold.to(device)

        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(is_train):
                logits, loss, _ =  model(x, y, scaffold)
                loss = loss.mean()
                losses.append(loss.item())

        if is_train:
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            if config.lr_decay:
                tokens += (y >= 0).sum()
                if tokens < config.warmup_tokens:
                    lr_mult = float(tokens) / float(max(1, config.warmup_tokens))
                else:
                    progress = float(tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config.learning_rate
            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

    if is_train:
        return float(np.mean(losses)), tokens

    if not is_train:
        test_loss = float(np.mean(losses))
        return test_loss, tokens

def get_NLLs_batch_transformer(smiles_list,transformer_model, vocab,stoi, regex, max_len, scaffold_max_len, batch_size=200, device='cpu'):
    """Batch process NLL calculator for Transformer model using DataLoader."""
    
    smiles_list=utils.filter_large_smiles(smiles_list,max_len-2)
    scaffold = utils.extract_scaffold(smiles_list)

    lens = [len(regex.findall(i.strip())) for i in (smiles_list)]
    new_max_len = max(lens)+2

    lens = [len(regex.findall(i.strip())) for i in (scaffold)]
    new_scaffold_max_len = max(lens)+2
    
    if new_max_len > max_len:
        raise ValueError("SMILES length exceeds the model's maximum allowable length.")
        
    if new_scaffold_max_len > scaffold_max_len:
        raise ValueError("SCAFFOLD length exceeds the model's maximum allowable length.")
        
    smiles_list = ['[SOS]' + i + '[EOS]' for i in smiles_list]
 
    smiles_list = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles_list]    
    scaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
    
    dataset = SmileDataset(
        debug=False,
        data=smiles_list,
        content=vocab,
        block_size=max_len,
        aug_prob=0,
        scaffold=scaffold,
        scaffold_maxlen=scaffold_max_len
    )

    dataloader = DataLoader(dataset, shuffle=False,batch_size=batch_size,num_workers=10 )
    
    nlls_list = []
    pad_token_id = stoi['<']  
    transformer_model.eval()
    
    with torch.no_grad():
        for batch_idx, (x, y, sca_tensor) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            sca_tensor = sca_tensor.to(device)
            logits, _, _ = transformer_model(x, y, sca_tensor)
            log_probs = F.log_softmax(logits, dim=-1)

            logits_flat = log_probs.view(-1, log_probs.size(-1))  
            target_flat = y.view(-1) 

            mask = target_flat != pad_token_id
            nll_loss = F.nll_loss(logits_flat, target_flat, reduction='none')
            nll_loss = nll_loss * mask.float()

            nll_per_sample = nll_loss.view(x.size(0), -1).sum(dim=1)
            nlls_list.append(nll_per_sample.cpu().numpy())
            
    nlls_cat = np.concatenate(nlls_list) if nlls_list else np.array([])
    return nlls_cat   

def train_config_to_dict(config):
    return {
        'max_epochs': config.max_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'lr_decay': config.lr_decay,
        'warmup_tokens': config.warmup_tokens,
        'final_tokens': config.final_tokens,
        'num_workers': config.num_workers,
        'save_ckpt': config.save_ckpt,
        'block_size': config.block_size
    }

def fine_tuning_run(split, model, train_dataset, 
              optimizer, scaler, device, epoch, 
              config, tokens,  stoi, itos):
    is_train = split == 'train'
    model.train(is_train)
    
    data = train_dataset 
    loader = DataLoader(data, shuffle=True, pin_memory=True,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers)
    
    losses = []
    pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
    
    for it, (x, y, scaffold) in pbar:
        x = x.to(device)
        y = y.to(device)
        scaffold = scaffold.to(device)

        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(is_train):
                logits, loss, _ =  model(x, y, scaffold)
                loss = loss.mean()
                losses.append(loss.item())

        if is_train:
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            if config.lr_decay:
                tokens += (y >= 0).sum()
                if tokens < config.warmup_tokens:
                    lr_mult = float(tokens) / float(max(1, config.warmup_tokens))
                else:
                    progress = float(tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config.learning_rate

            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

    if is_train:
        return float(np.mean(losses)), tokens

    if not is_train:
        test_loss = float(np.mean(losses))
        return test_loss, tokens