"""
Treinamento do BiSheafNodeClassifier em classificacao de nos.

Rode com:
    python -m src.experiments.bisheaf_diffusion.train_node
    python -m src.experiments.bisheaf_diffusion.train_node --dataset Cora
    python -m src.experiments.bisheaf_diffusion.train_node --dataset Texas --split 3
    python -m src.experiments.bisheaf_diffusion.train_node --dataset Texas --all-splits
    python -m src.experiments.bisheaf_diffusion.train_node --save-dir runs/bisheaf_diffusion_texas
"""

import argparse
import copy
import json
import math
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.utils import to_undirected
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.BiSheafDiffusion import BiSheafNodeClassifier, compute_reverse_idx


PLANETOID_DATASETS   = {"cora", "citeseer", "pubmed"}
WEBKB_DATASETS       = {"cornell", "texas", "wisconsin"}
WIKIPEDIA_DATASETS   = {"squirrel", "chameleon"}
MULTISPLIT_DATASETS  = WEBKB_DATASETS | WIKIPEDIA_DATASETS | {"film"}


# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------

def load_data(dataset_name: str, root: str = "data", split_idx: int = 0):
    name_lower = dataset_name.lower()

    if name_lower in WEBKB_DATASETS:
        dataset = WebKB(root=root, name=dataset_name)
        data    = dataset[0]
    elif name_lower in WIKIPEDIA_DATASETS:
        dataset = WikipediaNetwork(root=os.path.join(root, "WikipediaNetwork"), name=dataset_name)
        data    = dataset[0]
    elif name_lower == "film":
        dataset = Actor(root=os.path.join(root, "Film"))
        data    = dataset[0]
    else:
        dataset = Planetoid(root=root, name=dataset_name)
        data    = dataset[0]

    if name_lower in MULTISPLIT_DATASETS:
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask   = data.val_mask[:, split_idx]
        data.test_mask  = data.test_mask[:, split_idx]

    data.edge_index  = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.reverse_idx = compute_reverse_idx(data.edge_index)
    return data, dataset.num_classes


def load_data_shared(dataset_name: str, root: str = "data"):
    """
    Carrega dataset sem selecionar split e move todos os tensores para
    shared memory do PyTorch. Usar no processo pai antes de lançar workers:
    o dataset fica em memória uma única vez, compartilhado sem cópia.

    Retorna (data, num_classes).
    - Datasets multi-split (WebKB, Wikipedia, Film): masks são 2D [n, 10].
    - Planetoid: masks são 1D (split único).
    Usar _apply_split() nos workers para selecionar o split correto.
    """
    name_lower = dataset_name.lower()

    if name_lower in WEBKB_DATASETS:
        dataset = WebKB(root=root, name=dataset_name)
        data    = dataset[0]
    elif name_lower in WIKIPEDIA_DATASETS:
        dataset = WikipediaNetwork(root=os.path.join(root, "WikipediaNetwork"), name=dataset_name)
        data    = dataset[0]
    elif name_lower == "film":
        dataset = Actor(root=os.path.join(root, "Film"))
        data    = dataset[0]
    else:
        dataset = Planetoid(root=root, name=dataset_name)
        data    = dataset[0]

    data.edge_index  = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.reverse_idx = compute_reverse_idx(data.edge_index)

    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, torch.Tensor):
            data[key] = val.share_memory_()

    return data, dataset.num_classes


def _apply_split(base_data, split_idx: int, dataset_name: str):
    """
    Cria cópia rasa do base_data (tensores grandes ainda compartilhados)
    e substitui as masks pelo slice do split correto (clone privado).
    """
    import copy as _copy
    data = _copy.copy(base_data)
    if dataset_name.lower() in MULTISPLIT_DATASETS:
        data.train_mask = base_data.train_mask[:, split_idx].clone()
        data.val_mask   = base_data.val_mask[:, split_idx].clone()
        data.test_mask  = base_data.test_mask[:, split_idx].clone()
    return data


# ---------------------------------------------------------------------------
# Treino e avaliacao
# ---------------------------------------------------------------------------

def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index, data.reverse_idx)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    if not torch.isfinite(loss):
        return loss.item(), 0.0
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        optimizer.zero_grad()
    pred      = out.detach().argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    return loss.item(), train_acc


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out  = model(data.x, data.edge_index, data.reverse_idx)
    pred = out.argmax(dim=1)
    return (pred[mask] == data.y[mask]).float().mean().item()


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(save_dir=None, cfg_override=None, quiet=False, preloaded_data=None, num_classes_hint=None):
    from src.experiments.bisheaf_diffusion.config_node import CONFIG

    cfg = copy.deepcopy(CONFIG)
    if cfg_override:
        for k, v in cfg_override.items():
            if k == "model" and isinstance(v, dict):
                cfg["model"] = {**cfg["model"], **v}
            else:
                cfg[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    dataset_name  = cfg["dataset"]
    split_idx     = cfg.get("split_idx", 0)
    is_multisplit = dataset_name.lower() in MULTISPLIT_DATASETS
    split_str     = f" [split {split_idx}]" if is_multisplit else ""

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  BiSheafNodeClassifier — {dataset_name}{split_str}")
        print(f"{'='*60}\n")

    if preloaded_data is not None:
        data        = _apply_split(preloaded_data, split_idx, dataset_name)
        num_classes = num_classes_hint
    else:
        data, num_classes = load_data(dataset_name, split_idx=split_idx)
    data = data.to(device)

    node_feat_dim = data.x.shape[1]
    if not quiet:
        print(f"  nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}")
        print(f"  node_feat={node_feat_dim}  classes={num_classes}")
        print(f"  train={data.train_mask.sum().item()}  "
              f"val={data.val_mask.sum().item()}  "
              f"test={data.test_mask.sum().item()}\n")

    mcfg  = cfg["model"]
    model = BiSheafNodeClassifier(
        node_feat_dim=node_feat_dim,
        d=mcfg["d"],
        hidden_channels=mcfg["hidden_channels"],
        num_layers=mcfg["num_layers"],
        num_classes=num_classes,
        learner_hidden=mcfg.get("learner_hidden", 64),
        learner_gnn_layers=mcfg.get("learner_gnn_layers", 1),
        dropout=mcfg.get("dropout", 0.0),
        left_weights=mcfg.get("left_weights", True),
        right_weights=mcfg.get("right_weights", True),
        use_eps=mcfg.get("use_eps", True),
        norm_eps=mcfg.get("norm_eps", 1e-3),
        mlp_dims=mcfg.get("mlp_dims", None),
        backbone_F=mcfg.get("backbone_F", "node_edge_node"),
        backbone_C=mcfg.get("backbone_C", "node_edge_node"),
        map_type_F=mcfg.get("map_type_F", "general"),
        map_type_C=mcfg.get("map_type_C", "general"),
        orth_trans=mcfg.get("orth_trans", "householder"),
        sheaf_act=mcfg.get("sheaf_act", "id"),
        inter_layer_norm=mcfg.get("inter_layer_norm", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if not quiet:
        print(f"  Parametros: {n_params:,}\n")

    train_on_trainval = cfg.get("train_on_trainval", False)
    if train_on_trainval:
        data.train_mask = data.train_mask | data.val_mask
        if not quiet:
            print(f"  [train_on_trainval=True] mask efetiva: {data.train_mask.sum().item()} nos\n")

    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_weights  = None
    print_every   = cfg.get("print_every", 50)
    patience      = cfg.get("patience", 0)
    patience_ctr  = 0
    nan_streak    = 0
    epochs_run    = cfg["epochs"]
    t0            = time.time()

    pbar = tqdm(range(cfg["epochs"]), desc="Training", leave=True, disable=quiet)
    for epoch in pbar:
        loss, train_acc = train_epoch(model, data, optimizer, criterion)

        if math.isnan(loss):
            nan_streak += 1
            if nan_streak >= 10:
                if not quiet:
                    tqdm.write(f"  Early stop: NaN loss por {nan_streak} epochs consecutivos (epoch {epoch+1})")
                epochs_run = epoch + 1
                break
        else:
            nan_streak = 0

        test_acc = evaluate(model, data, data.test_mask)

        if train_on_trainval:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            if not quiet:
                pbar.set_postfix_str(
                    f"loss={loss:.3f} train={train_acc:.3f} test={test_acc:.3f} best={best_test_acc:.3f}"
                )
                if (epoch + 1) % print_every == 0:
                    tqdm.write(
                        f"  Ep {epoch+1:4d}/{cfg['epochs']}"
                        f" | loss={loss:.4f} | train={train_acc:.4f}"
                        f" | test={test_acc:.4f} | best={best_test_acc:.4f}"
                        f" | {time.time()-t0:.0f}s"
                    )
        else:
            val_acc = evaluate(model, data, data.val_mask)
            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_test_acc = test_acc
                best_weights  = copy.deepcopy(model.state_dict())
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if not quiet:
                pbar.set_postfix_str(
                    f"loss={loss:.3f} train={train_acc:.3f} val={val_acc:.3f} best_val={best_val_acc:.3f} test@best={best_test_acc:.3f}"
                )
                if (epoch + 1) % print_every == 0:
                    tqdm.write(
                        f"  Ep {epoch+1:4d}/{cfg['epochs']}"
                        f" | loss={loss:.4f} | train={train_acc:.4f} | val={val_acc:.4f}"
                        f" | best_val={best_val_acc:.4f} | test@best={best_test_acc:.4f}"
                        f" | {time.time()-t0:.0f}s"
                    )

            if patience > 0 and patience_ctr >= patience:
                if not quiet:
                    tqdm.write(f"  Early stop: epoch {epoch+1}")
                epochs_run = epoch + 1
                break

    total_time = time.time() - t0

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  [{dataset_name}] Resultado final")
        if train_on_trainval:
            print(f"  Best test acc:  {best_test_acc:.4f}")
        else:
            print(f"  Best val acc:   {best_val_acc:.4f}")
            print(f"  Test@best val:  {best_test_acc:.4f}")
        print(f"  {epochs_run} epochs  |  {total_time:.1f}s")
        print(f"{'='*60}")

    if save_dir and best_weights is not None:
        torch.save(best_weights, os.path.join(save_dir, "best_model.pt"))
        with open(os.path.join(save_dir, "results.json"), "w") as fh:
            json.dump({
                "best_val_acc": best_val_acc,
                "test_at_best_val": best_test_acc,
                "epochs_run": epochs_run,
                "total_time_s": round(total_time, 2),
                "config": cfg,
            }, fh, indent=2)

    return {
        "best_val_acc": best_val_acc,
        "test_at_best_val": best_test_acc,
        "epochs_run": epochs_run,
        "total_time_s": round(total_time, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, default=None)
    parser.add_argument("--save-dir",   type=str, default=None)
    parser.add_argument("--split",      type=int, default=None)
    parser.add_argument("--all-splits", action="store_true",
                        help="Roda os 10 splits WebKB e reporta media +- std")
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    if args.all_splits:
        import numpy as np
        from src.experiments.bisheaf_diffusion.config_node import CONFIG
        dataset_name = override.get("dataset", CONFIG["dataset"])
        if dataset_name.lower() not in MULTISPLIT_DATASETS:
            print(f"AVISO: --all-splits é para datasets com múltiplos splits. Dataset: {dataset_name}")
        results = []
        for s in range(10):
            print(f"\n--- Split {s}/9 ---")
            r = main(cfg_override={**override, "split_idx": s}, quiet=True)
            results.append(r["test_at_best_val"])
            print(f"    test@best_val = {r['test_at_best_val']:.4f}")
        arr = np.array(results) * 100
        print(f"\n{'='*60}")
        print(f"  {dataset_name} — 10 splits")
        print(f"  {[f'{v:.1f}' for v in arr]}")
        print(f"  Media: {arr.mean():.2f}%  ±  {arr.std():.2f}%")
        print(f"{'='*60}")
    else:
        if args.split is not None:
            override["split_idx"] = args.split
        main(save_dir=args.save_dir, cfg_override=override or None)
