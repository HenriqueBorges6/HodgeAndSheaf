"""
Treinamento 10-fold CV para CoSheafNetwork em datasets TUDataset.

Rode com:
    python -m src.experiments.cosheaf.train
    python -m src.experiments.cosheaf.train --dataset PTC_MR
    python -m src.experiments.cosheaf.train --dataset ENZYMES --save-dir results/cosheaf/enzymes
"""

import argparse
import copy
import json
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree as pyg_degree
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from torch_geometric.data import Batch

from src.experiments.cosheaf.config import CONFIG
from src.experiments.cosheaf.edge_features import augment_edge_features
from src.models.CoSheafNN import CoSheafNetwork, build_line_graph_edge_index


# ---------------------------------------------------------------------------
# Carregamento e normalizacao do dataset
# ---------------------------------------------------------------------------

def load_data(dataset_name: str = "MUTAG", root: str = "data", edge_feat_mode: str = "none"):
    raw = TUDataset(root=f"{root}/{dataset_name}", name=dataset_name)
    dataset = list(raw)

    # Re-indexa edge_index para indices locais 0-based por grafo
    # (necessario para PROTEINS e ENZYMES que podem ter indices nao contiguos)
    for data in dataset:
        if data.edge_index.numel() == 0:
            continue
        unique_nodes, new_edge_index = torch.unique(data.edge_index, return_inverse=True)
        new_edge_index = new_edge_index.reshape(2, -1)
        n_unique = unique_nodes.shape[0]

        if data.x is not None and data.x.shape[0] != n_unique:
            x_expanded = torch.zeros(n_unique, data.x.shape[1], dtype=data.x.dtype)
            old_n = data.x.shape[0]
            mask = unique_nodes < old_n
            x_expanded[mask] = data.x[unique_nodes[mask]]
            data.x = x_expanded

        data.edge_index = new_edge_index

    # Features de no: usa grau se ausente
    for data in dataset:
        if data.x is None:
            n = data.edge_index.max().item() + 1 if data.edge_index.numel() > 0 else 0
            deg = pyg_degree(data.edge_index[1], num_nodes=n).unsqueeze(-1).float()
            data.x = deg

    # Detecta se o dataset tem features de aresta nativas
    has_native = any(
        d.edge_attr is not None and d.edge_attr.shape[0] == d.edge_index.shape[1]
        for d in dataset
    )

    if not has_native and edge_feat_mode != "none":
        # Augmenta com features artificiais (estruturais e/ou derivadas dos nós)
        for data in dataset:
            data.edge_attr = augment_edge_features(data, mode=edge_feat_mode)
        in_edge_dim = dataset[0].edge_attr.shape[1]
    else:
        # Comportamento padrão: ones se ausente (preserva features nativas se existirem)
        in_edge_dim = 1
        for data in dataset:
            if data.edge_attr is not None and data.edge_attr.shape[0] == data.edge_index.shape[1]:
                in_edge_dim = data.edge_attr.shape[1]
                break
        for data in dataset:
            if data.edge_attr is None or data.edge_attr.shape[0] != data.edge_index.shape[1]:
                data.edge_attr = torch.ones(data.edge_index.shape[1], in_edge_dim)

    # Pré-computa grafo de linhas por grafo — evita recalcular a cada forward
    for data in dataset:
        num_nodes = data.x.shape[0]
        lei, ls = build_line_graph_edge_index(num_nodes, data.edge_index)
        data.line_edge_index = lei   # (2, E_L)
        data.line_signs      = ls    # (E_L,)

    labels      = [data.y.item() for data in dataset]
    in_node_dim = dataset[0].x.shape[1]
    num_classes = int(max(labels)) + 1

    return dataset, labels, in_node_dim, in_edge_dim, num_classes


# ---------------------------------------------------------------------------
# Collate: concatena grafos e corrige offsets do grafo de linhas
# ---------------------------------------------------------------------------

def collate_with_line_graph(data_list):
    """
    Collate function que:
    1. Deixa o PyG batchear normalmente (edge_index com offset de nós, etc.)
    2. Concatena line_edge_index com offset correto de arestas por grafo.

    Necessário porque line_edge_index indexa arestas do grafo original,
    não nós — o DataLoader padrão do PyG não aplica esse offset.
    """
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

def _forward(model, batch):
    return model(batch.x, batch.edge_attr, batch.edge_index, batch.batch,
                 batch.line_edge_index, batch.line_signs)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = _forward(model, batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pred = out.detach().argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total   += batch.y.size(0)
    avg_loss  = total_loss / len(loader)
    train_acc = correct / total if total > 0 else 0.0
    return avg_loss, train_acc


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = _forward(model, batch)
            pred  = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total   += batch.y.size(0)
    return correct / total


def profile_model(model, loader, device, n_warmup=3, n_runs=10):
    """Roda forward passes com profiling e retorna media dos tempos por etapa."""
    model.eval()
    sample = next(iter(loader)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _forward(model, sample)

    # Profiling
    model.set_profiling(True)
    from collections import defaultdict
    accum = defaultdict(float)
    with torch.no_grad():
        for _ in range(n_runs):
            _forward(model, sample)
            for k, v in model.get_profile().items():
                accum[k] += v
    model.set_profiling(False)

    return {k: round(v / n_runs * 1000, 3) for k, v in accum.items()}  # ms


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(save_dir=None, cfg_override=None):
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

    dataset_name   = cfg["dataset"]
    edge_feat_mode = cfg.get("edge_feat_mode", "none")
    dataset, labels, in_node_dim, in_edge_dim, num_classes = load_data(
        dataset_name, edge_feat_mode=edge_feat_mode
    )

    cfg["model"]["node_feat_dim"] = in_node_dim
    cfg["model"]["edge_feat_dim"] = in_edge_dim
    cfg["model"]["num_classes"]   = num_classes

    skf       = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg["seed"])
    criterion = nn.CrossEntropyLoss()
    fold_accs  = []
    fold_times = []
    total_epochs = 0
    profile_data = None
    print_every  = cfg.get("print_every", 50)

    run_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        train_set = [dataset[i] for i in train_idx]
        test_set  = [dataset[i] for i in test_idx]

        pin = device.type == "cuda"
        train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                                  pin_memory=pin, collate_fn=collate_with_line_graph)
        test_loader  = DataLoader(test_set,  batch_size=cfg["batch_size"], shuffle=False,
                                  pin_memory=pin, collate_fn=collate_with_line_graph)

        model     = CoSheafNetwork(**cfg["model"]).to(device)
        optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # Profiling na primeira fold (antes de treinar)
        if fold == 0:
            profile_data = profile_model(model, train_loader, device)
            tqdm.write(f"\n  [Profile] Tempos medios por forward pass (ms):")
            for k, v in sorted(profile_data.items()):
                tqdm.write(f"    {k:<25s} {v:>8.3f} ms")
            tqdm.write("")

        best_acc     = 0.0
        best_weights = None
        fold_start   = time.time()

        pbar = tqdm(range(cfg["epochs"]), desc=f"Fold {fold+1:2d}", leave=False)
        for epoch in pbar:
            loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, test_loader, device)

            if val_acc > best_acc:
                best_acc     = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            pbar.set_postfix_str(
                f"loss={loss:.3f} train={train_acc:.3f} val={val_acc:.3f} best={best_acc:.3f}"
            )

            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - fold_start
                tqdm.write(
                    f"  [Fold {fold+1:2d}] Ep {epoch+1:3d}/{cfg['epochs']}"
                    f" | loss={loss:.4f} | train={train_acc:.4f}"
                    f" | val={val_acc:.4f} | best={best_acc:.4f}"
                    f" | {elapsed:.0f}s"
                )

        fold_elapsed = time.time() - fold_start
        fold_accs.append(best_acc)
        fold_times.append(fold_elapsed)
        total_epochs += cfg["epochs"]
        tqdm.write(f"  Fold {fold+1:2d} concluido: best={best_acc:.4f} ({fold_elapsed:.1f}s)")

        if save_dir:
            torch.save(best_weights, os.path.join(save_dir, f"fold_{fold+1}_model.pt"))

    total_time     = time.time() - run_start
    avg_fold_time  = sum(fold_times) / len(fold_times)
    avg_epoch_time = total_time / total_epochs if total_epochs > 0 else 0.0

    mean_acc = sum(fold_accs) / len(fold_accs)
    std_acc  = (sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)) ** 0.5

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  [{dataset_name}] Acuracia: {mean_acc:.4f} +/- {std_acc:.4f}")
    tqdm.write(f"  Folds: {[f'{a:.4f}' for a in fold_accs]}")
    tqdm.write(f"  Tempo total: {total_time:.1f}s | por fold: {avg_fold_time:.1f}s | por epoch: {avg_epoch_time:.3f}s")
    tqdm.write(f"{'='*60}")

    if save_dir:
        results = {
            "fold_accs":        fold_accs,
            "mean":             mean_acc,
            "std":              std_acc,
            "total_time_s":     round(total_time, 2),
            "avg_fold_time_s":  round(avg_fold_time, 2),
            "avg_epoch_time_s": round(avg_epoch_time, 4),
            "fold_times_s":     [round(t, 2) for t in fold_times],
            "profile_ms":       profile_data,
            "config":           cfg,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return {
        "mean": mean_acc, "std": std_acc, "fold_accs": fold_accs,
        "total_time_s": round(total_time, 2),
        "avg_fold_time_s": round(avg_fold_time, 2),
        "avg_epoch_time_s": round(avg_epoch_time, 4),
        "profile_ms": profile_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str, default=None,
                        help="Nome do dataset TUDataset (default: valor em config.py)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Diretorio para salvar pesos e resultados")
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    main(save_dir=args.save_dir, cfg_override=override or None)