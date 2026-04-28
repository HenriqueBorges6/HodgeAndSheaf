"""
Treinamento do SheafNN em datasets Planetoid (Cora, Citeseer, PubMed).

Rode com:
    python -m src.experiments.sheaf_nn.train
    python -m src.experiments.sheaf_nn.train --dataset Citeseer
    python -m src.experiments.sheaf_nn.train --conv general
"""

import argparse
import copy
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import GCNConv
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.SheafNN import SheafNN


PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
WEBKB_DATASETS = {"cornell", "texas", "wisconsin"}


class GCN(nn.Module):
    """Baseline GCN 2 camadas para comparacao."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def load_data(dataset_name: str = "Cora", root: str = "data", split_idx: int = 0):
    """
    Carrega dataset para node classification.
    Suporta Planetoid (Cora, Citeseer, PubMed) e WebKB (Texas, Cornell, Wisconsin).

    WebKB tem 10 splits fixos armazenados como (n, 10) masks.
    split_idx seleciona qual dos 10 splits usar.
    """
    name_lower = dataset_name.lower()
    if name_lower in WEBKB_DATASETS:
        dataset = WebKB(root=root, name=dataset_name)
        data = dataset[0]
        # WebKB masks shape: (n, 10) — selecionar split
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask   = data.val_mask[:, split_idx]
        data.test_mask  = data.test_mask[:, split_idx]
        return data, dataset.num_classes
    else:
        dataset = Planetoid(root=root, name=dataset_name)
        return dataset[0], dataset.num_classes


# ---------------------------------------------------------------------------
# Treino / avaliacao
# ---------------------------------------------------------------------------

def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
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
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return (pred[mask] == data.y[mask]).float().mean().item()


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main(cfg_override=None, use_gcn_baseline=False):
    from src.experiments.sheaf_nn.config import CONFIG

    cfg = copy.deepcopy(CONFIG)
    if cfg_override:
        for k, v in cfg_override.items():
            if k == "model" and isinstance(v, dict):
                cfg["model"] = {**cfg["model"], **v}
            else:
                cfg[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    dataset_name = cfg["dataset"]
    split_idx = cfg.get("split_idx", 0)
    mcfg = cfg["model"]

    data, num_classes = load_data(dataset_name, split_idx=split_idx)
    data = data.to(device)
    node_feat_dim = data.x.shape[1]

    split_str = f" [split {split_idx}]" if dataset_name.lower() in WEBKB_DATASETS else ""
    if use_gcn_baseline:
        print(f"\n{'='*60}")
        print(f"  GCN Baseline — {dataset_name}{split_str}")
        print(f"{'='*60}\n")
        print(f"  nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}")
        print(f"  train={data.train_mask.sum().item()}"
              f"  val={data.val_mask.sum().item()}"
              f"  test={data.test_mask.sum().item()}\n")
        model = GCN(node_feat_dim, 64, num_classes, dropout=0.5).to(device)
    else:
        print(f"\n{'='*60}")
        print(f"  SheafNN [{mcfg['conv_type']}] — {dataset_name}{split_str}")
        print(f"{'='*60}\n")
        print(f"  nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}")
        print(f"  node_feat={node_feat_dim}  classes={num_classes}")
        print(f"  train={data.train_mask.sum().item()}"
              f"  val={data.val_mask.sum().item()}"
              f"  test={data.test_mask.sum().item()}\n")
        model = SheafNN(
            node_feat_dim=node_feat_dim,
            d=mcfg["d"],
            hidden_channels=mcfg["hidden_channels"],
            num_layers=mcfg["num_layers"],
            num_classes=num_classes,
            conv_type=mcfg.get("conv_type", "bundle"),
            orth_trans=mcfg.get("orth_trans", "matrix_exp"),
            gnn_type=mcfg.get("gnn_type", "SAGE"),
            gnn_layers=mcfg.get("gnn_layers", 1),
            gnn_hidden=mcfg.get("gnn_hidden", 64),
            dropout=mcfg.get("dropout", 0.6),
            left_weights=mcfg.get("left_weights", True),
            right_weights=mcfg.get("right_weights", True),
            use_eps=mcfg.get("use_eps", True),
            mlp_dims=mcfg.get("mlp_dims", None),
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,}\n")

    # train_on_trainval: treina com train+val, avalia so no test (diagnostico de data starvation)
    train_on_trainval = cfg.get("train_on_trainval", False)
    if train_on_trainval:
        data.train_mask = data.train_mask | data.val_mask
        print(f"  [train_on_trainval=True] mask efetiva: {data.train_mask.sum().item()} nos\n")

    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    print_every = cfg.get("print_every", 50)
    train_start = time.time()

    pbar = tqdm(range(cfg["epochs"]), desc="Training", leave=True)
    for epoch in pbar:
        loss, train_acc = train_epoch(model, data, optimizer, criterion)
        test_acc = evaluate(model, data, data.test_mask)

        if train_on_trainval:
            # Sem val separado — acompanhar melhor test diretamente
            if test_acc > best_test_acc:
                best_test_acc = test_acc
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

            gap = train_acc - val_acc
            pbar.set_postfix_str(
                f"loss={loss:.3f} train={train_acc:.3f} val={val_acc:.3f} "
                f"gap={gap:+.3f} best_val={best_val_acc:.3f}"
            )
            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - train_start
                tqdm.write(
                    f"  Ep {epoch+1:3d}/{cfg['epochs']}"
                    f" | loss={loss:.4f} | train={train_acc:.4f}"
                    f" | val={val_acc:.4f} | test={test_acc:.4f}"
                    f" | gap={gap:+.4f}"
                    f" | best_val={best_val_acc:.4f} (test@best={best_test_acc:.4f})"
                    f" | {elapsed:.0f}s"
                )

    total_time = time.time() - train_start

    print(f"\n{'='*60}")
    print(f"  [{dataset_name}] Resultado final")
    if train_on_trainval:
        print(f"  [train_on_trainval=True]")
        print(f"  Best test acc: {best_test_acc:.4f}")
    else:
        print(f"  Best val acc:  {best_val_acc:.4f}")
        print(f"  Test@best val: {best_test_acc:.4f}")
    print(f"  Tempo: {total_time:.1f}s total | {total_time/cfg['epochs']:.3f}s/epoch")
    print(f"{'='*60}")

    return {"best_val_acc": best_val_acc, "test_at_best_val": best_test_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--conv", type=str, default=None, help="'bundle' ou 'general'")
    parser.add_argument("--baseline", action="store_true", help="Rodar GCN baseline")
    parser.add_argument("--split", type=int, default=None, help="Split index para WebKB (0-9)")
    parser.add_argument("--all-splits", action="store_true", help="Rodar todos os 10 splits WebKB e reportar media +- std")
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset
    if args.conv:
        override["model"] = {"conv_type": args.conv}

    if args.all_splits:
        import numpy as np
        from src.experiments.sheaf_nn.config import CONFIG
        dataset_name = override.get("dataset", CONFIG["dataset"])
        if dataset_name.lower() not in WEBKB_DATASETS:
            print(f"AVISO: --all-splits so faz sentido para WebKB (Texas, Cornell, Wisconsin). Dataset atual: {dataset_name}")
        results = []
        for s in range(10):
            print(f"\n--- Split {s}/9 ---")
            r = main(cfg_override={**override, "split_idx": s}, use_gcn_baseline=args.baseline)
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
        main(cfg_override=override or None, use_gcn_baseline=args.baseline)
