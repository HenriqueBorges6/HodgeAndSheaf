"""
Treinamento da CoSheafNodeNetwork (Delta_1 + readout aresta->no)
em datasets Planetoid (Cora, Citeseer, PubMed).

Rode com:
    python -m src.experiments.cosheaf_node.train
    python -m src.experiments.cosheaf_node.train --dataset Citeseer
    python -m src.experiments.cosheaf_node.train --dataset PubMed --save-dir runs/cosheaf_node_pubmed
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
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.experiments.cosheaf.edge_features import augment_edge_features
from src.models.CoSheafNode import CoSheafNodeNetwork
from src.models.CoSheafNN.utils import build_line_graph_edge_index


# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------

def load_data(dataset_name: str = "Cora", root: str = "data",
              edge_feat_mode: str = "structural", conv_mode: str = "direct"):
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]

    # Gerar edge features (Planetoid nao tem edge_attr)
    if edge_feat_mode != "none":
        data.edge_attr = augment_edge_features(data, mode=edge_feat_mode)
    else:
        data.edge_attr = torch.ones(data.edge_index.shape[1], 1)

    edge_feat_dim = data.edge_attr.shape[1]

    # Pre-computar grafo de linhas (so necessario para conv_mode='linegraph')
    if conv_mode == 'linegraph':
        num_nodes = data.x.shape[0]
        lei, ls = build_line_graph_edge_index(num_nodes, data.edge_index)
        data.line_edge_index = lei
        data.line_signs = ls

    return data, edge_feat_dim, dataset.num_classes


# ---------------------------------------------------------------------------
# Treino e avaliacao
# ---------------------------------------------------------------------------

def _get_line_graph(data):
    """Retorna (line_edge_index, line_signs) ou (None, None) se nao pre-computado."""
    if hasattr(data, 'line_edge_index'):
        return data.line_edge_index, data.line_signs
    return None, None


def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    lei, ls = _get_line_graph(data)
    out = model(data.x, data.edge_index, lei, ls)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    pred = out.detach().argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    return loss.item(), train_acc


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    lei, ls = _get_line_graph(data)
    out = model(data.x, data.edge_index, lei, ls)
    pred = out.argmax(dim=1)
    return (pred[mask] == data.y[mask]).float().mean().item()


def profile_model(model, data, device, n_warmup=3, n_runs=10):
    """Roda forward passes com profiling e retorna media dos tempos por etapa."""
    model.eval()
    from collections import defaultdict

    with torch.no_grad():
        for _ in range(n_warmup):
            model(data.x, data.edge_index, *_get_line_graph(data))

    model.set_profiling(True)
    accum = defaultdict(float)
    with torch.no_grad():
        for _ in range(n_runs):
            model(data.x, data.edge_index, *_get_line_graph(data))
            for k, v in model.get_profile().items():
                accum[k] += v
    model.set_profiling(False)

    return {k: round(v / n_runs * 1000, 3) for k, v in accum.items()}  # ms


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(save_dir=None, cfg_override=None):
    from src.experiments.cosheaf_node.config import CONFIG

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
    edge_feat_mode = cfg.get("edge_feat_mode", "structural")

    print(f"\n{'='*60}")
    print(f"  CoSheafNodeNetwork — {dataset_name}")
    print(f"  edge_feat_mode: {edge_feat_mode}")
    print(f"{'='*60}\n")

    conv_mode = cfg["model"].get("conv_mode", "direct")

    # Carregar dados
    data, edge_feat_dim, num_classes = load_data(
        dataset_name, edge_feat_mode=edge_feat_mode, conv_mode=conv_mode)
    data = data.to(device)

    node_feat_dim = data.x.shape[1]
    has_lg = hasattr(data, 'line_edge_index') and data.line_edge_index is not None
    line_edges_str = str(data.line_edge_index.shape[1]) if has_lg else "N/A (direct)"
    print(f"  nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}"
          f"  line_edges={line_edges_str}  conv_mode={conv_mode}")
    print(f"  node_feat={node_feat_dim}  edge_feat={edge_feat_dim}  classes={num_classes}")
    print(f"  train={data.train_mask.sum().item()}  val={data.val_mask.sum().item()}"
          f"  test={data.test_mask.sum().item()}")
    print()

    # Construir modelo
    model = CoSheafNodeNetwork(
        node_feat_dim=node_feat_dim,
        d=cfg["model"]["d"],
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=num_classes,
        learner_hidden=cfg["model"]["learner_hidden"],
        mlp_dims=cfg["model"].get("mlp_dims"),
        dropout=cfg["model"].get("dropout", 0.0),
        backbone=cfg["model"].get("backbone", "sage"),
        orth_method=cfg["model"].get("orth_method", "cayley"),
        conv_mode=conv_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}")

    # Profiling
    profile_data = profile_model(model, data, device)
    print(f"\n  [Profile] Tempos medios por forward pass (ms):")
    for k, v in sorted(profile_data.items()):
        print(f"    {k:<25s} {v:>8.3f} ms")
    print()

    # Treino
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_weights = None
    print_every = cfg.get("print_every", 20)
    train_start = time.time()

    pbar = tqdm(range(cfg["epochs"]), desc="Training", leave=True)
    for epoch in pbar:
        loss, train_acc = train_epoch(model, data, optimizer, criterion)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())

        pbar.set_postfix_str(
            f"loss={loss:.3f} train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f} best_val={best_val_acc:.3f}"
        )

        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - train_start
            tqdm.write(
                f"  Ep {epoch+1:3d}/{cfg['epochs']}"
                f" | loss={loss:.4f} | train={train_acc:.4f}"
                f" | val={val_acc:.4f} | test={test_acc:.4f}"
                f" | best_val={best_val_acc:.4f} (test@best={best_test_acc:.4f})"
                f" | {elapsed:.0f}s"
            )

    total_time = time.time() - train_start
    avg_epoch_time = total_time / cfg["epochs"] if cfg["epochs"] > 0 else 0

    print(f"\n{'='*60}")
    print(f"  [{dataset_name}] Resultado final")
    print(f"  Best val acc:  {best_val_acc:.4f}")
    print(f"  Test@best val: {best_test_acc:.4f}")
    print(f"  Tempo: {total_time:.1f}s total | {avg_epoch_time:.3f}s/epoch")
    print(f"{'='*60}")

    if save_dir:
        torch.save(best_weights, os.path.join(save_dir, "best_model.pt"))
        results = {
            "best_val_acc": best_val_acc,
            "test_at_best_val": best_test_acc,
            "total_time_s": round(total_time, 2),
            "avg_epoch_time_s": round(avg_epoch_time, 4),
            "profile_ms": profile_data,
            "config": cfg,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return {
        "best_val_acc": best_val_acc,
        "test_at_best_val": best_test_acc,
        "total_time_s": round(total_time, 2),
        "avg_epoch_time_s": round(avg_epoch_time, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    main(save_dir=args.save_dir, cfg_override=override or None)
