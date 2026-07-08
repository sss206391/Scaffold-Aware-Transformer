import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.nn import Linear, ReLU, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.nn import GATConv, GCNConv, global_max_pool as gmp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TrainConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    grad_norm_clip = 1.0
    weight_decay = 0.1 
    lr_decay = False
    save_ckpt = None
    num_workers = 0
    betas = (0.9, 0.95)
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class GATConfig:
    def __init__(self, num_features, n_filters, embed_dim, output_dim, dropout, num_heads,concat, n_output=1, **kwargs):
        self.num_features = num_features
        self.n_filters = n_filters
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.concat = concat
        self.n_output = 1
        for k, v in kwargs.items():
            setattr(self, k, v)

class GATNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config=config

        self.gcn1 = GATConv(config.num_features, config.embed_dim,
                            config.num_heads, config.dropout, config.concat)
        self.gcn2 = GATConv(config.embed_dim * config.num_heads , config.embed_dim,
                            config.num_heads, config.dropout, config.concat)
        self.gcn3 = GATConv(config.embed_dim * config.num_heads, config.output_dim,
                            config.num_heads, config.dropout, config.concat)
        self.fc_g1 = Linear(config.output_dim*config.num_heads, config.output_dim)

        self.fc1 = Linear(config.output_dim, 64)
        self.fc2 = Linear(64, 32)
        self.out = Linear(32, config.n_output)

        self.relu = ReLU()
        self.dropout = Dropout(config.dropout)
        
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
                elif isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif isinstance(m, GATConv) and (pn.endswith('lin_src.weight') or pn.endswith('lin_dst.weight')):
                    decay.add(fpn)
                elif isinstance(m, GATConv):
                    no_decay.add(fpn)

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

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x, attn1 = self.gcn1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        
        x, attn2 = self.gcn2(x, edge_index, return_attention_weights=True)  
        x = F.elu(x)
        x, attn3 = self.gcn3(x, edge_index, return_attention_weights=True)
        x = self.relu(x)
        x = gmp(x, batch)          
        x = self.fc_g1(x)
        x = self.relu(x)

        xc = self.fc1(x)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out, attn1, attn2, attn3  

def train(split,model, train_loader,test_loader, optimizer, scaler, device, epoch, config, criterion):
    logger = logging.getLogger(__name__)
    is_train=split=='train'
    
    model.train(is_train)
    
    if is_train==True:    
        epoch_loss = 0
        pbar=tqdm(enumerate(train_loader), total=len(train_loader))
        for it, data in pbar:
            data = data.to(device)
            model.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):                
                    output,_,_,_ = model(data)
                    loss = criterion(output, data.y)
                    epoch_loss += loss.item()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

            lr=config.learning_rate

            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {epoch_loss:.5f}. lr {lr:e}")

        return epoch_loss / len(train_loader)
    
    elif is_train==False:
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output,_,_,_ = model(data)
                loss = criterion(output, data.y)
                total_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        logger.info("test loss: %f", total_loss)

        return total_loss / len(test_loader), mse, r2
    
    else:
        print("error")
        
def test(model, test_loader, device, criterion):
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar=tqdm(enumerate(test_loader), total=len(test_loader))
    
    with torch.no_grad():
        for it, data in pbar:
            data = data.to(device)
            output,_,_,_ = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return total_loss / len(test_loader), mse, r2