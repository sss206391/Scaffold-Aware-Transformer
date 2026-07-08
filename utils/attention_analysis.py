import os
import io
from PIL import Image, ImageDraw, ImageFont
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from . import chemistry

def _wrap_text_to_width(text, font, max_width):
    draw = ImageDraw.Draw(Image.new("RGB", (10,10)))
    lines, buf = [], ""
    for ch in text:
        w, _ = draw.textsize(buf + ch, font=font)
        if w <= max_width:
            buf += ch
        else:
            if buf: lines.append(buf)
            buf = ch
    if buf: lines.append(buf)
    return lines

def _draw_captioned_highlight(mol, highlight_atoms, highlight_bonds,
                              out_path, smiles_text=None,     
                              img_width=1000, base_height=560,
                              font_scale=1.4, atom_radius=0.45, bond_width=3):
    from rdkit.Chem.Draw import rdMolDraw2D
    show_caption = bool(smiles_text)          
    margin = 16
    pad_bottom = 0
    if show_caption:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", int(18 * font_scale))
        except Exception:
            font = ImageFont.load_default()
        max_caption_width = img_width - 2*margin
        lines = _wrap_text_to_width(smiles_text, font, max_caption_width)
        line_h = int(22 * font_scale)
        pad_bottom = margin + line_h * len(lines) + margin


    core_h = base_height                        
    pmol   = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(img_width, core_h)
    dopts  = drawer.drawOptions()
    dopts.addStereoAnnotation = False
    dopts.fixedBondLength     = 38              
    dopts.padding             = 0.01            

    col = (0.95, 0.20, 0.20)
    atom_cols  = {int(a): col for a in highlight_atoms}
    bond_cols  = {int(b): col for b in highlight_bonds}
    atom_rads  = {int(a): float(atom_radius) for a in highlight_atoms}
    bond_wmult = {int(b): float(bond_width)  for b in highlight_bonds}

    try:
        drawer.DrawMoleculeWithHighlights(
            pmol, "", atom_cols, bond_cols, atom_rads, bond_wmult, -1
        )
    except TypeError:
        drawer.DrawMolecule(pmol, legend="",
                            highlightAtoms=list(map(int, highlight_atoms)),
                            highlightBonds=list(map(int, highlight_bonds)))
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    mol_img   = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    canvas = Image.new("RGB", (img_width, core_h + pad_bottom), "white")
    canvas.paste(mol_img, (0, 0))

    if show_caption:
        draw = ImageDraw.Draw(canvas)
        y = core_h + margin
        for ln in lines:
            draw.text((margin, y), ln, fill=(0,0,0), font=font)
            y += line_h

    canvas.save(out_path)
    return out_path

def _to_batch(obj):
    if isinstance(obj, Data):
        return Batch.from_data_list([obj])
    if isinstance(obj, dict):
        return Batch.from_data_list([Data(**obj)])
    if isinstance(obj, (list, tuple)):
        dl = []
        for it in obj:
            if isinstance(it, Data): dl.append(it)
            elif isinstance(it, dict): dl.append(Data(**it))
            else: raise RuntimeError(f"Unsupported featurizer type: {type(it)}")
        if not dl: raise RuntimeError("Featurizer returned an empty list")
        return Batch.from_data_list(dl)
    raise RuntimeError(f"Unsupported return type from featurizer: {type(obj)}")

def _parse_attn_tuple(attn):
    if isinstance(attn, tuple) and len(attn) == 2:
        eidx, alpha = attn
        eidx  = torch.as_tensor(eidx, dtype=torch.long, device='cpu')
        alpha = torch.as_tensor(alpha, dtype=torch.float32, device='cpu')
        if alpha.dim() == 2: alpha = alpha.mean(1)
        elif alpha.dim() != 1: alpha = alpha.flatten()
        return eidx, alpha
    if isinstance(attn, (list, tuple)) and attn and isinstance(attn[0], tuple):
        e0, a0 = _parse_attn_tuple(attn[0])
        alphas = [a0] + [_parse_attn_tuple(t)[1] for t in attn[1:]]
        return e0, torch.stack(alphas, 0).mean(0)
    raise RuntimeError("Failed to parse attention format")

def _manual_submol(parent_mol, atom_ids, bond_ids):
    atom_ids = list(dict.fromkeys(int(a) for a in atom_ids))  # uniq & keep order
    amap = {aid: i for i, aid in enumerate(atom_ids)}
    em = Chem.RWMol(Chem.Mol())

    for aid in atom_ids:
        a = parent_mol.GetAtomWithIdx(aid)
        na = Chem.Atom(a.GetAtomicNum())
        em.AddAtom(na)

    for bid in bond_ids:
        b = parent_mol.GetBondWithIdx(int(bid))
        i = amap.get(b.GetBeginAtomIdx())
        j = amap.get(b.GetEndAtomIdx())
        if i is not None and j is not None:
            em.AddBond(i, j, b.GetBondType())
    m = em.GetMol()
    Chem.SanitizeMol(m, catchErrors=True)
    return m

def _build_submol_safe(parent_mol, atom_ids, bond_ids):

    atom_ids = sorted(set(int(a) for a in atom_ids))
    bond_ids = sorted(set(int(b) for b in bond_ids))

    submol = None
    if bond_ids:
        try:

            submol = Chem.PathToSubmol(parent_mol, bond_ids)  
            Chem.SanitizeMol(submol, catchErrors=True)
        except Exception:
            submol = _manual_submol(parent_mol, atom_ids, bond_ids)
    else:
        submol = _manual_submol(parent_mol, atom_ids, bond_ids)

    try:
        sub_smiles = Chem.MolToSmiles(submol, canonical=True)
    except Exception:
        sub_smiles = Chem.MolToSmiles(submol, canonical=False)
    return submol, sub_smiles


def _largest_component(node_mask, edge_index):
    src, dst = edge_index
    N = int(max(src.max().item(), dst.max().item())) + 1
    adj = [[] for _ in range(N)]
    for i, j in zip(src.tolist(), dst.tolist()):
        if node_mask[i] and node_mask[j]:
            adj[i].append(j); adj[j].append(i)
    seen = [False]*N; best=[]
    for v in range(N):
        if not node_mask[v] or seen[v]: continue
        comp, q = [], [v]; seen[v]=True
        while q:
            u=q.pop(); comp.append(u)
            for w in adj[u]:
                if node_mask[w] and not seen[w]:
                    seen[w]=True; q.append(w)
        if len(comp)>len(best): best=comp
    mask = torch.zeros(N, dtype=torch.bool)
    if best: mask[best] = True
    return mask

def _expand_one_hop(edge_index, node_ids):
    src, dst = edge_index
    mask = torch.zeros(int(max(src.max().item(), dst.max().item()))+1, dtype=torch.bool)
    mask[node_ids] = True
    touched = mask[src] | mask[dst]
    mask[src[touched]] = True
    mask[dst[touched]] = True
    return torch.nonzero(mask, as_tuple=False).view(-1).tolist()

def _edge_score_from_attn(attn_list):
    ref_eidx = None
    ref_keys = None
    AL = []

    K = 10_000_007  
    for a in attn_list:
        eidx, alpha = _parse_attn_tuple(a)  
        keys = (eidx[0] * K + eidx[1]).tolist()

        if ref_eidx is None:
            ref_eidx = eidx
            ref_keys = keys
            AL.append(alpha)
        else:
            if len(keys) != len(ref_keys):
                raise RuntimeError("Edge sets differ across layers; cannot compute mean attention")
            pos = {k: i for i, k in enumerate(keys)}
            perm = [pos[k] for k in ref_keys]  
            alpha = alpha[perm]
            AL.append(alpha)

    s = torch.stack(AL, 0).mean(0)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return ref_eidx, s

def _node_score_from_edges(edge_index, s_edge, reduce="mean"):
    src, dst = edge_index
    N = int(max(src.max().item(), dst.max().item())) + 1
    s_node = torch.zeros(N, dtype=torch.float32)
    if reduce in ("mean", "sum"):
        agg = torch.zeros(N, dtype=torch.float32)
        cnt = torch.zeros(N, dtype=torch.float32)
        for u, v, s in zip(src.tolist(), dst.tolist(), s_edge.tolist()):
            agg[u] += s; agg[v] += s
            cnt[u] += 1; cnt[v] += 1
        if reduce == "mean":
            s_node = agg / cnt.clamp_min(1)
        else:
            s_node = agg
    elif reduce == "max":
        s_node[:] = 0.0
        for u, v, s in zip(src.tolist(), dst.tolist(), s_edge.tolist()):
            s_node[u] = max(float(s_node[u]), float(s))
            s_node[v] = max(float(s_node[v]), float(s))
    else:
        raise ValueError("reduce must be one of: mean, sum, or max")
    s_node = (s_node - s_node.min()) / (s_node.max() - s_node.min() + 1e-9)
    return s_node

def _node_score_from_attn(attn_node_list, N=None):
    NL = []
    for a in (attn_node_list if isinstance(attn_node_list, (list, tuple)) else [attn_node_list]):
        a = torch.as_tensor(a, dtype=torch.float32, device="cpu")
        if a.dim() == 2:  
            a = a.mean(1)
        elif a.dim() != 1:
            a = a.flatten()
        NL.append(a if N is None else a[:N])
    s = torch.stack(NL, 0).mean(0)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s

def Attention_analysis(
    smiles, predictor, featurizer, save_dir, prefix="MOL",

    select_mode="hybrid",              
    top_percent_edges=0.15,             
    top_percent_nodes=0.15,           
    node_weight_direct=0.5,             
    node_weight_from_edge=0.5,          
    node_reduce_from_edges="mean",      
    require_node_attn=False,          
    expand_hops=1,
    img_width=1000, base_height=560, font_scale=1.4,
    atom_radius=0.45, bond_width=3,
    largest_cc=True
):
    os.makedirs(save_dir, exist_ok=True)

    try:
        check_ok, can = chemistry.check_and_standardize(smiles)
        base_smiles = can if (check_ok and can) else smiles
    except Exception:
        base_smiles = smiles.split('.')[0]

    try:
        obj = featurizer([base_smiles])
    except TypeError:
        obj = featurizer(base_smiles)
    batch = _to_batch(obj)
    device = next(predictor.parameters()).device
    batch  = batch.to(device)

    ret = predictor(batch)
    if isinstance(ret, tuple):
        out = ret[0]
        attn_list = list(ret[1:])
        node_attn_list = None
    elif isinstance(ret, dict):
        out = ret.get("out", ret.get("pred", ret.get("logits", None)))
        attn_list = ret.get("attn_list", ret.get("attn", None))
        node_attn_list = (
            ret.get("node_attn_list", None)
            or ret.get("node_attn", None)
            or ret.get("node_alpha", None)
            or ret.get("node_attention", None)
        )
    else:
        out = ret
        attn_list = getattr(predictor, "last_attn", None)
        node_attn_list = None

    if attn_list is None:
        raise RuntimeError("Failed to obtain edge attention from predictor")

    edge_index, s_edge = _edge_score_from_attn(attn_list) 
    src, dst = edge_index
    N_nodes = int(max(src.max().item(), dst.max().item())) + 1

    s_node_direct = None
    if node_attn_list is None:
        node_attn_list = getattr(predictor, "last_node_attn", None)
    if node_attn_list is not None:
        s_node_direct = _node_score_from_attn(node_attn_list, N=N_nodes) 

    s_node_from_edge = _node_score_from_edges(edge_index, s_edge, reduce=node_reduce_from_edges)  

    core_nodes = None
    bonds_to_hl = []

    if select_mode == "edge":
        thresh_e = torch.quantile(s_edge, 1 - float(top_percent_edges)).item()
        mask_e = s_edge >= thresh_e
        core_nodes = torch.unique(torch.cat([src[mask_e], dst[mask_e]])).tolist()

    else:
        if select_mode == "node":
            if s_node_direct is None:
                if require_node_attn:
                    raise RuntimeError("Node attention is required but was not provided")
                s_node = s_node_from_edge
            else:
                s_node = s_node_direct
        elif select_mode == "hybrid":
            sources = []
            weights = []
            if s_node_direct is not None and node_weight_direct > 0:
                sources.append(s_node_direct)
                weights.append(float(node_weight_direct))
            if s_node_from_edge is not None and node_weight_from_edge > 0:
                sources.append(s_node_from_edge)
                weights.append(float(node_weight_from_edge))
            if not sources:
                sources = [s_node_from_edge]
                weights = [1.0]
            w = torch.tensor(weights, dtype=torch.float32)
            w = w / (w.sum() + 1e-9)
            s_node = sum(w[i] * sources[i] for i in range(len(sources)))
        else:
            raise ValueError("select_mode must be one of: edge, node, or hybrid")

        thresh_n = torch.quantile(s_node, 1 - float(top_percent_nodes)).item()
        node_mask = s_node >= thresh_n
        core_nodes = torch.nonzero(node_mask, as_tuple=False).view(-1).tolist()

    nodes_expanded = core_nodes
    for _ in range(max(0, int(expand_hops))):
        nodes_expanded = _expand_one_hop(edge_index, nodes_expanded)

    node_mask = torch.zeros(N_nodes, dtype=torch.bool)
    node_mask[nodes_expanded] = True

    if largest_cc:
        node_mask = _largest_component(node_mask, edge_index)
    core_idx = torch.nonzero(node_mask, as_tuple=False).view(-1).tolist()
    core_set = set(core_idx)

    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        raise ValueError("MolFromSmiles failed: " + base_smiles)
    bonds_to_hl = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in core_set and j in core_set:
            bonds_to_hl.append(b.GetIdx())

    sub_mol, sub_smiles = None, ""
    if core_idx:
        sub_mol, sub_smiles = _build_submol_safe(mol, core_idx, bonds_to_hl)

    img1 = os.path.join(save_dir, f"{prefix}.png")
    img2 = os.path.join(save_dir, f"{prefix}_sub.png")

    _draw_captioned_highlight(
        mol, core_idx, bonds_to_hl, img1,
        smiles_text=None,               
        img_width=2000, base_height=1200,
        font_scale=1.2, atom_radius=0.45, bond_width=4
    )

    try:
        pred_value = float(torch.as_tensor(out).view(-1)[0].detach().cpu())
    except Exception:
        pred_value = None

    result = {
        "smiles": base_smiles,
        "pred": pred_value,
        "sub_smiles": sub_smiles,
        "highlight_atoms": core_idx,
        "highlight_bonds": bonds_to_hl,
        "image_main": img1,
        "image_sub": img2,
        "selection_mode": select_mode,
        "top_percent_edges": top_percent_edges,
        "top_percent_nodes": top_percent_nodes,
        "expand_hops": expand_hops,
    }

    result["edge_scores"] = s_edge.detach().cpu().tolist()
    if select_mode == "edge":
        result["node_scores"] = _node_score_from_edges(edge_index, s_edge, reduce=node_reduce_from_edges).tolist()
    else:
        if select_mode == "node" and s_node_direct is not None:
            result["node_scores"] = s_node.detach().cpu().tolist()
        elif select_mode == "hybrid":
            result["node_scores"] = s_node.detach().cpu().tolist()
        else:
            result["node_scores"] = _node_score_from_edges(edge_index, s_edge, reduce=node_reduce_from_edges).tolist()

    return result