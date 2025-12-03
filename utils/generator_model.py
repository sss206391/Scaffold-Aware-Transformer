import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from utils import utils
from utils.dataset import SmileDataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TransformerConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, embd_pdrop=0.1,
                 resid_pdrop=0.1, attn_pdrop=0.1, weight_decay=0.1, learning_rate=0.001,
                 betas=(0.9, 0.95), scaffold_maxlen=100, scaffold_weight=2.0, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas
        self.scaffold_maxlen = scaffold_maxlen
        self.scaffold_weight = scaffold_weight  
        for k, v in kwargs.items():
            setattr(self, k, v)

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.scaffold_weight = nn.Parameter(torch.tensor(config.scaffold_weight * 0.1))

        
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + config.scaffold_maxlen, 
                                                           config.block_size + config.scaffold_maxlen))
                             .view(1, 1, config.block_size + config.scaffold_maxlen, 
                                   config.block_size + config.scaffold_maxlen))

    def forward(self, x, scaffold_mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if scaffold_mask is not None:
            att = att + scaffold_mask * self.scaffold_weight     
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att_score = att
        
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att_score

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, scaffold_mask=None):
        y, attn = self.attn(self.ln1(x), scaffold_mask)
        
        x = x + y
        x = x + self.mlp(self.ln2(x))
                        
        return x, attn

def focal_loss(logits, targets, alpha=1, gamma=2, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss  # Focal Loss formula
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
    
class MultiScaleScaffoldAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scaffold_attn = nn.ModuleList([
            nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.attn_pdrop)
            for _ in range(3)  
        ])
        self.scales = [1, 2, 4] 
        self.proj = nn.Linear(config.n_embd, config.n_embd) 

    def forward(self, scaffold_embeddings):
        # scaffold_embeddings: [batch_size, scaffold_length, embedding_dim]
        scaffold_embeddings = scaffold_embeddings.permute(1, 0, 2)  # [scaffold_length, batch_size, embedding_dim]

        multi_scale_attn_outputs = []
        for scale, attn in zip(self.scales, self.scaffold_attn):
            
            scaled_scaffold = scaffold_embeddings[::scale] 
            attn_output, _ = attn(scaled_scaffold, scaled_scaffold, scaled_scaffold)
            
            attn_output_resized = F.interpolate(attn_output.permute(1, 2, 0), size=scaffold_embeddings.size(0)).permute(2, 0, 1)
            multi_scale_attn_outputs.append(attn_output_resized)

        combined_attn_output = torch.mean(torch.stack(multi_scale_attn_outputs), dim=0)  # [scaffold_length, batch_size, embd_dim]
        
        combined_attn_output = self.proj(combined_attn_output.permute(1, 0, 2))  # [batch_size, scaffold_length, embd_dim]
        
        return combined_attn_output

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.scaffold_attention = MultiScaleScaffoldAttention(config)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.block_size = config.block_size
        self.scaffold_maxlen = config.scaffold_maxlen
        
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size       
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        for mn, m in self.named_modules():
            if hasattr(m, 'scaffold_weight'):
                no_decay.add(f'{mn}.scaffold_weight')
        
        
        for i, attn_layer in enumerate(self.scaffold_attention.scaffold_attn):
            for pn, p in attn_layer.named_parameters():
                fpn = f'scaffold_attention.scaffold_attn.{i}.{pn}'
                if 'weight' in pn and 'bias' not in pn:
                    decay.add(fpn) 
                else:
                    no_decay.add(fpn) 

        
        no_decay.add('pos_emb')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, scaffold=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        
        scaffold_mask = None

        if scaffold is not None:
            scaffold_embeddings = self.tok_emb(scaffold) 
            #[batch_size, scaffold_length, embedding_dim]
            
            scaffold_attn_output = self.scaffold_attention(scaffold_embeddings) 
            
            scaffold_position_embeddings = self.pos_emb[:, :self.scaffold_maxlen, :]
            # [1,scaffold_maxlen, embedding_dim]
            
            scaffold_x = self.drop(scaffold_attn_output + scaffold_position_embeddings)
            # [batch_size, scaffold_maxlen, embedding_dim]
            
            x = torch.cat((scaffold_x, x), dim=1) 
            # [batchsize, scaffold_maxlen + sequence_length, embedding_dim]
            
            scaffold_mask = torch.zeros((b, 1,  t + self.scaffold_maxlen, t + self.scaffold_maxlen), device=x.device)
            # [batch_size, 1 , scaffold_maxlen + sequence_length, scaffold_maxlen + sequence_length]            
            
            scaffold_mask[:, :, :self.scaffold_maxlen, :self.scaffold_maxlen] = 1
            scaffold_mask[:, :, self.scaffold_maxlen:, self.scaffold_maxlen:] = 1

        attn_maps = []
                
        for layer in self.blocks:
            x, attn = layer(x, scaffold_mask) 
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        
        if self.config.scaffold:
            num = int(self.config.scaffold_maxlen) 
        else:
            num = 0
        
        logits = logits[:, num:, :]

        loss = None
        
        if targets is not None:
            # Reshape logits and targets for focal loss
            logits = logits.reshape(-1, logits.size(-1))  # (batch_size * sequence_length, vocab_size)
            targets = targets.reshape(-1)  # (batch_size * sequence_length)

            loss = focal_loss(logits, targets, alpha=0.25, gamma=1, reduction='mean')

        return logits, loss, attn_maps


def get_NLLs_batch_transformer(smiles_list,transformer_model, vocab,stoi, regex, max_len, scaffold_max_len, batch_size=200, device='cpu'):
    """Batch process NLL calculator for Transformer model using DataLoader."""
    
    smiles_list=new_utils.filter_large_smiles(smiles_list,max_len-2)
    scaffold = new_utils.extract_scaffold(smiles_list)

    print("NLL start")
    print("smiles_list",len(smiles_list))
    print("scaffold",len(scaffold))
    
    lens = [len(regex.findall(i.strip())) for i in (smiles_list)]
    new_max_len = max(lens)+2
    print('new_Max len: ', new_max_len)

    lens = [len(regex.findall(i.strip())) for i in (scaffold)]
    new_scaffold_max_len = max(lens)+2
    print('new_scaffold_max_len: ', new_scaffold_max_len)
    
    
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