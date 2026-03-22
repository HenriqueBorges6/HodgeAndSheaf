"""
Treinamento com splits oficiais OGB para CoSheafNetwork.

Datasets suportados:
  ogbg-molhiv  — binario, metrica: ROC-AUC
  ogbg-ppa     — 37 classes, metrica: accuracy

Uso:
    python -m src.experiments.cosheaf.train_ogb
    python -m src.experiments.cosheaf.train_ogb --dataset ogbg-ppa
    python -m src.experiments.cosheaf.train_ogb --dataset ogbg-molhiv --save-dir results/cosheaf/molhiv
"""

import argparse
import copy
import json
import os
import warnings

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree as pyg_degree
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.CoSheafNN import CoSheafNetwork, build_line_graph_edge_index


# ---------------------------------------------------------------------------
# Metadados por dataset
# ---------------------------------------------------------------------------

DATASET_META = {
    "ogbg-molhiv": {"num_classes": 1,  "loss": "bce", "metric": "rocauc"},
    "ogbg-ppa":    {"num_classes": 37, "loss": "ce",  "metric": "acc"},
}

CONFIG_OGB = {
    "dataset":      "ogbg-molhiv",
    "batch_size":   128,
    "epochs":       100,
    "seed":         1,
    "lr":           0.001,
    "weight_decay": 1e-4,
    "model": {
        # node_feat_dim, edge_feat_dim e num_classes detectados automaticamente
        "d":              4,
        "hidden_dims":    [64, 64],
        "learner_hidden": [32],
        "mlp_dims":       [128],
        "dropout":        0.25,
    },
}


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_data(dataset_name: str, root: str = "data"):
    raw       = PygGraphPropPredDataset(name=dataset_name, root=root)
    split_idx = raw.get_idx_split()
    dataset   = list(raw)

    # ogbg-ppa nao tem features de no — usa grau como substituto
    for data in dataset:
        if data.x is None:
            deg    = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes)
            data.x = deg.unsqueeze(-1).float()

    # Garante que y seja 1-D por grafo: (1,) em vez de (1, 1)
    for data in dataset:
        data.y = data.y.flatten()

    # Pre-computa grafo de linhas (evita recalcular a cada forward)
    for data in dataset:
        lei, ls          = build_line_graph_edge_index(data.x.shape[0], data.edge_index)
        data.line_edge_index = lei
        data.line_signs      = ls

    in_node_dim = dataset[0].x.shape[1]
    in_edge_dim = dataset[0].edge_attr.shape[1]
    num_classes = DATASET_META[dataset_name]["num_classes"]

    train_set = [dataset[i] for i in split_idx["train"]]
    val_set   = [dataset[i] for i in split_idx["valid"]]
    test_set  = [dataset[i] for i in split_idx["test"]]

    return train_set, val_set, test_set, in_node_dim, in_edge_dim, num_classes


# ---------------------------------------------------------------------------
# Collate: mesma logica do train.py, concatena line_edge_index com offsets
# ---------------------------------------------------------------------------

def collate_with_line_graph(data_list):
    batch = Batch.from_data_list(data_list, exclude_keys=["line_edge_index", "line_signs"])

    edge_counts  = torch.tensor([d.edge_index.shape[1] for d in data_list])
    edge_offsets = torch.zeros_like(edge_counts)
    edge_offsets[1:] = edge_counts[:-1].cumsum(0)

    lei_parts, ls_parts = [], []
    for offset, d in zip(edge_offsets.tolist(), data_list):
        if d.line_edge_index.shape[1] > 0:
            lei_parts.append(d.line_edge_index + int(offset))
            ls_parts.append(d.line_signs)

    if lei_parts:
        batch.line_edge_index = torch.cat(lei_parts, dim=1)
        batch.line_signs      = torch.cat(ls_parts)
    else:
        batch.line_edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch.line_signs      = torch.zeros(0)

    return batch


# ---------------------------------------------------------------------------
# Treino e avaliacao
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, meta, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch,
                     batch.line_edge_index, batch.line_signs)

        if meta["loss"] == "bce":
            y = batch.y.float()
            is_labeled = ~torch.isnan(y)
            loss = criterion(out.view(-1)[is_labeled], y[is_labeled])
        else:
            loss = criterion(out, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, meta, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch,
                          batch.line_edge_index, batch.line_signs)

            if meta["metric"] == "rocauc":
                y_true_all.append(batch.y.cpu())
                y_pred_all.append(out.view(-1).sigmoid().cpu())
            else:
                y_true_all.append(batch.y.cpu())
                y_pred_all.append(out.argmax(dim=1).cpu())

    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)

    if meta["metric"] == "rocauc":
        mask = ~torch.isnan(y_true)
        return roc_auc_score(y_true[mask].numpy(), y_pred[mask].numpy())
    else:
        return (y_pred == y_true).float().mean().item()


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(save_dir=None, cfg_override=None):
    cfg = copy.deepcopy(CONFIG_OGB)
    if cfg_override:
        for k, v in cfg_override.items():
            if k == "model" and isinstance(v, dict):
                cfg["model"] = {**cfg["model"], **v}
            else:
                cfg[k] = v

    dataset_name = cfg["dataset"]
    meta         = DATASET_META[dataset_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    train_set, val_set, test_set, in_node_dim, in_edge_dim, num_classes = load_data(dataset_name)

    cfg["model"]["node_feat_dim"] = in_node_dim
    cfg["model"]["edge_feat_dim"] = in_edge_dim
    cfg["model"]["num_classes"]   = num_classes

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              pin_memory=pin, collate_fn=collate_with_line_graph)
    val_loader   = DataLoader(val_set,   batch_size=cfg["batch_size"], shuffle=False,
                              pin_memory=pin, collate_fn=collate_with_line_graph)
    test_loader  = DataLoader(test_set,  batch_size=cfg["batch_size"], shuffle=False,
                              pin_memory=pin, collate_fn=collate_with_line_graph)

    model     = CoSheafNetwork(**cfg["model"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss() if meta["loss"] == "bce" else nn.CrossEntropyLoss()

    best_val     = 0.0
    best_weights = None

    for epoch in tqdm(range(cfg["epochs"]), desc=dataset_name):
        loss       = train_epoch(model, train_loader, optimizer, criterion, meta, device)
        val_metric = evaluate(model, val_loader, meta, device)

        if val_metric > best_val:
            best_val     = val_metric
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"  Epoch {epoch+1:3d} | loss: {loss:.4f} | val: {val_metric:.4f} | best: {best_val:.4f}")

    # Avalia no test com melhor checkpoint (escolhido por val)
    model.load_state_dict(best_weights)
    test_metric = evaluate(model, test_loader, meta, device)

    metric = meta["metric"]
    tqdm.write(f"\n[{dataset_name}] val_{metric}={best_val:.4f}  test_{metric}={test_metric:.4f}")

    if save_dir:
        torch.save(best_weights, os.path.join(save_dir, "best_model.pt"))
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump({"val": best_val, "test": test_metric, "metric": metric, "config": cfg},
                      f, indent=2)

    return {"val": best_val, "test": test_metric, "metric": metric}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    main(save_dir=args.save_dir, cfg_override=override or None)
