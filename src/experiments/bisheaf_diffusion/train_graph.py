"""
Treinamento do BiSheafGraphClassifier em classificacao de grafos (TUDataset).

10-fold cross-validation estratificado, igual ao HodgeGNN.

Rode com:
    python -m src.experiments.bisheaf_diffusion.train_graph
    python -m src.experiments.bisheaf_diffusion.train_graph --dataset PROTEINS
    python -m src.experiments.bisheaf_diffusion.train_graph --save-dir runs/bisheaf_diffusion_mutag
"""

import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.BiSheafDiffusion import BiSheafGraphClassifier


# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------

def load_dataset(name: str, root: str = "data"):
    dataset = TUDataset(root=os.path.join(root, name), name=name)

    # Datasets sem features de no (ex: IMDB-B): usa grau como feature escalar
    if dataset[0].x is None:
        max_degree = 0
        for data in dataset:
            deg = data.edge_index[0].bincount(minlength=data.num_nodes)
            max_degree = max(max_degree, deg.max().item())
        # Injeta one-hot do grau como feature
        import torch
        from torch_geometric.transforms import OneHotDegree
        dataset = TUDataset(
            root=os.path.join(root, name), name=name,
            transform=OneHotDegree(max_degree),
        )

    # Remove grafos degenerados: sem nós ou sem arestas causam operações vazias
    valid   = [data for data in dataset if data.num_nodes > 0 and data.edge_index.shape[1] > 0]
    removed = len(dataset) - len(valid)
    if removed:
        print(f"  [{name}] {removed} grafo(s) degenerado(s) removidos")
    dataset = valid
    labels  = [data.y.item() for data in dataset]
    return dataset, labels


# ---------------------------------------------------------------------------
# Treino e avaliacao por fold
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        pred   = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total   += batch.y.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(save_dir=None, cfg_override=None, quiet=False):
    from src.experiments.bisheaf_diffusion.config_graph import CONFIG

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

    dataset_name = cfg["dataset"]

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  BiSheafGraphClassifier — {dataset_name}  ({cfg['n_folds']}-fold CV)")
        print(f"{'='*60}\n")

    dataset, labels = load_dataset(dataset_name)
    node_feat_dim   = dataset[0].x.shape[1]
    num_classes     = dataset.num_classes

    if not quiet:
        print(f"  grafos={len(dataset)}  classes={num_classes}  node_feat={node_feat_dim}")
        nos_media = np.mean([d.num_nodes for d in dataset])
        print(f"  nos/grafo (media): {nos_media:.1f}\n")

    mcfg = cfg["model"]
    skf  = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    criterion = nn.CrossEntropyLoss()

    fold_accs = []
    t0_total  = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        train_loader = DataLoader(
            [dataset[i] for i in train_idx],
            batch_size=cfg["batch_size"], shuffle=True,
        )
        test_loader = DataLoader(
            [dataset[i] for i in test_idx],
            batch_size=cfg["batch_size"],
        )

        model = BiSheafGraphClassifier(
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
            pooling=mcfg.get("pooling", "sum"),
            mlp_dims=mcfg.get("mlp_dims", None),
        ).to(device)

        optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        print_every = cfg.get("print_every", 50)
        t0_fold = time.time()

        pbar = tqdm(range(cfg["epochs"]), desc=f"Fold {fold+1}/{cfg['n_folds']}", disable=quiet)
        last_acc = 0.0
        for epoch in pbar:
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            if not quiet:
                if (epoch + 1) % print_every == 0:
                    last_acc = evaluate(model, test_loader, device)
                    tqdm.write(
                        f"    Ep {epoch+1:3d}/{cfg['epochs']}"
                        f" | loss={loss:.4f} | test={last_acc:.4f}"
                        f" | {time.time()-t0_fold:.0f}s"
                    )
                    pbar.set_postfix_str(f"loss={loss:.3f} test={last_acc:.3f}")

        final_acc = evaluate(model, test_loader, device)
        fold_accs.append(final_acc)
        if not quiet:
            print(f"  → Fold {fold+1}: test={final_acc:.4f}  ({time.time()-t0_fold:.0f}s)\n")

    arr        = np.array(fold_accs) * 100
    total_time = time.time() - t0_total

    print(f"\n{'='*60}")
    print(f"  [{dataset_name}] {cfg['n_folds']}-fold CV — Resultado final")
    print(f"  Acc por fold: {[f'{v:.1f}' for v in arr]}")
    print(f"  Media: {arr.mean():.2f}%  ±  {arr.std():.2f}%")
    print(f"  {total_time:.1f}s total")
    print(f"{'='*60}")

    if save_dir:
        with open(os.path.join(save_dir, "results.json"), "w") as fh:
            json.dump({
                "dataset": dataset_name,
                "fold_accs": fold_accs,
                "mean_acc": float(arr.mean()),
                "std_acc": float(arr.std()),
                "total_time_s": round(total_time, 2),
                "config": cfg,
            }, fh, indent=2)

    return {
        "fold_accs": fold_accs,
        "mean_acc": float(arr.mean()),
        "std_acc": float(arr.std()),
        "total_time_s": round(total_time, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str, default=None,
                        help="Nome do TUDataset (MUTAG, PROTEINS, IMDB-B, ...)")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    main(save_dir=args.save_dir, cfg_override=override or None)
