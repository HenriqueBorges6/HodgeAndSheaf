"""
Treinamento do DualSheafDiffusion em datasets Planetoid (Cora, Citeseer, PubMed).

Rode com:
    python -m src.experiments.dual_sheaf.train
    python -m src.experiments.dual_sheaf.train --dataset Citeseer
    python -m src.experiments.dual_sheaf.train --dataset PubMed --save-dir runs/dual_sheaf_pubmed
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
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import to_undirected
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.DualSheaf import DualSheafDiffusion, compute_reverse_idx
from src.models.DualSheaf.dual_sheaf_network import compute_line_graph


PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
WEBKB_DATASETS = {"cornell", "texas", "wisconsin"}


# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------

def load_data(dataset_name: str = "Cora", root: str = "data", split_idx: int = 0):
    name_lower = dataset_name.lower()
    if name_lower in WEBKB_DATASETS:
        dataset = WebKB(root=root, name=dataset_name)
        data = dataset[0]
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask   = data.val_mask[:, split_idx]
        data.test_mask  = data.test_mask[:, split_idx]
        # WebKB pode ter arestas dirigidas sem reversa — garantir nao-direcionado
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    else:
        dataset = Planetoid(root=root, name=dataset_name)
        data = dataset[0]

    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.reverse_idx = compute_reverse_idx(data.edge_index)
    return data, dataset.num_classes


def ensure_line_graph(data):
    """Computa e cacheia o grafo de linhas em data.line_edge_index."""
    if not hasattr(data, 'line_edge_index') or data.line_edge_index is None:
        data.line_edge_index = compute_line_graph(data.edge_index, data.num_nodes)
    return data.line_edge_index


# ---------------------------------------------------------------------------
# Treino e avaliacao
# ---------------------------------------------------------------------------

def _forward(model, data):
    """Chama o forward passando line_edge_index apenas se cosheaf_b=True."""
    lei = data.line_edge_index if model.cosheaf_b else None
    return model(data.x, data.edge_index, data.reverse_idx, line_edge_index=lei)


def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = _forward(model, data)
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
    out = _forward(model, data)
    pred = out.argmax(dim=1)
    return (pred[mask] == data.y[mask]).float().mean().item()


def profile_model(model, data, n_warmup=3, n_runs=10):
    model.eval()
    from collections import defaultdict

    with torch.no_grad():
        for _ in range(n_warmup):
            _forward(model, data)

    model.set_profiling(True)
    accum = defaultdict(float)
    with torch.no_grad():
        for _ in range(n_runs):
            _forward(model, data)
            for k, v in model.get_profile().items():
                accum[k] += v
    model.set_profiling(False)

    return {k: round(v / n_runs * 1000, 3) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(save_dir=None, cfg_override=None, quiet=False):
    from src.experiments.dual_sheaf.config import CONFIG

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
    split_idx = cfg.get("split_idx", 0)
    split_str = f" [split {split_idx}]" if dataset_name.lower() in WEBKB_DATASETS else ""

    print(f"\n{'='*60}")
    print(f"  DualSheafDiffusion — {dataset_name}{split_str}")
    print(f"{'='*60}\n")

    # Carregar dados
    data, num_classes = load_data(dataset_name, split_idx=split_idx)
    data = data.to(device)

    node_feat_dim = data.x.shape[1]
    print(f"  nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}")
    print(f"  node_feat={node_feat_dim}  classes={num_classes}")
    print(f"  train={data.train_mask.sum().item()}  val={data.val_mask.sum().item()}"
          f"  test={data.test_mask.sum().item()}")
    print()

    # Construir modelo
    mcfg = cfg["model"]
    model = DualSheafDiffusion(
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
        single_learner=mcfg.get("single_learner", False),
        orth_trans=mcfg.get("orth_trans", None),
        laplacian_mode=mcfg.get("laplacian_mode", "dual"),
        cosheaf_b=mcfg.get("cosheaf_b", False),
        mlp_dims=mcfg.get("mlp_dims", None),
    ).to(device)

    # Pre-computar grafo de linhas se cosheaf_b=True
    if model.cosheaf_b:
        print("  [cosheaf_b=True] Computando grafo de linhas L(G)...")
        lei = ensure_line_graph(data)
        print(f"  L(G): {data.edge_index.shape[1]} nos, {lei.shape[1]} arestas\n")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}")

    # Profiling (suprimido em modo quiet)
    if not quiet:
        profile_data = profile_model(model, data)
        print(f"\n  [Profile] Tempos medios por forward pass (ms):")
        for k, v in sorted(profile_data.items()):
            print(f"    {k:<25s} {v:>8.3f} ms")
        print()
    else:
        profile_data = {}

    # train_on_trainval: treina com train+val, avalia so no test (diagnostico de data starvation)
    train_on_trainval = cfg.get("train_on_trainval", False)
    if train_on_trainval:
        data.train_mask = data.train_mask | data.val_mask
        print(f"  [train_on_trainval=True] mask efetiva: {data.train_mask.sum().item()} nos\n")

    # Treino
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_weights = None
    print_every = cfg.get("print_every", 20)
    patience = cfg.get("patience", 0)   # 0 = desabilitado
    patience_counter = 0
    epochs_run = cfg["epochs"]
    train_start = time.time()

    pbar = tqdm(range(cfg["epochs"]), desc="Training", leave=True, disable=quiet)
    for epoch in pbar:
        loss, train_acc = train_epoch(model, data, optimizer, criterion)
        test_acc = evaluate(model, data, data.test_mask)

        if train_on_trainval:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            if not quiet:
                pbar.set_postfix_str(
                    f"loss={loss:.3f} train={train_acc:.3f} test={test_acc:.3f} best_test={best_test_acc:.3f}"
                )
                if (epoch + 1) % print_every == 0:
                    elapsed = time.time() - train_start
                    tqdm.write(
                        f"  Ep {epoch+1:3d}/{cfg['epochs']}"
                        f" | loss={loss:.4f} | train={train_acc:.4f}"
                        f" | test={test_acc:.4f} | best_test={best_test_acc:.4f}"
                        f" | {elapsed:.0f}s"
                    )
        else:
            val_acc = evaluate(model, data, data.val_mask)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if not quiet:
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

            if patience > 0 and patience_counter >= patience:
                if not quiet:
                    tqdm.write(f"  Early stop: epoch {epoch+1} (patience={patience})")
                epochs_run = epoch + 1
                break

    total_time = time.time() - train_start
    avg_epoch_time = total_time / epochs_run if epochs_run > 0 else 0

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  [{dataset_name}] Resultado final")
        if train_on_trainval:
            print(f"  [train_on_trainval=True]")
            print(f"  Best test acc: {best_test_acc:.4f}")
        else:
            print(f"  Best val acc:  {best_val_acc:.4f}")
            print(f"  Test@best val: {best_test_acc:.4f}")
        print(f"  Epochs: {epochs_run}/{cfg['epochs']}  |  {total_time:.1f}s total | {avg_epoch_time:.3f}s/epoch")
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
        "epochs_run": epochs_run,
        "total_time_s": round(total_time, 2),
        "avg_epoch_time_s": round(avg_epoch_time, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--split", type=int, default=None, help="Split index para WebKB (0-9)")
    parser.add_argument("--all-splits", action="store_true", help="Rodar todos os 10 splits WebKB e reportar media +- std")
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    if args.all_splits:
        import numpy as np
        from src.experiments.dual_sheaf.config import CONFIG
        dataset_name = override.get("dataset", CONFIG["dataset"])
        if dataset_name.lower() not in WEBKB_DATASETS:
            print(f"AVISO: --all-splits so faz sentido para WebKB (Texas, Cornell, Wisconsin). Dataset atual: {dataset_name}")
        results = []
        for s in range(10):
            print(f"\n--- Split {s}/9 ---")
            r = main(cfg_override={**override, "split_idx": s})
            results.append(r["test_at_best_val"])
        arr = np.array(results) * 100
        print(f"\n{'='*60}")
        print(f"  {dataset_name} — 10 splits")
        print(f"  Test acc por split: {[f'{v:.1f}' for v in arr]}")
        print(f"  Media: {arr.mean():.2f}%  ±  {arr.std():.2f}%")
        print(f"{'='*60}")
    else:
        if args.split is not None:
            override["split_idx"] = args.split
        main(save_dir=args.save_dir, cfg_override=override or None)
