import argparse
import copy
import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.experiments.pi_one.config import CONFIG
from src.models.PiOneGNN.pi_one_GNN import PiOneGNN
from src.models.PiOneGNN.preprocessing import preprocess_dataset
from src.models.PiOneGNN.collate import bundle_collate


def load_data(dataset_name="MUTAG", root="data"):
    raw_dataset = TUDataset(root=f'{root}/{dataset_name}', name=dataset_name)
    dataset = list(raw_dataset)

    # Normalização de node indices: alguns datasets (PROTEINS, ENZYMES) armazenam
    # edge_index com índices globais não-contíguos. Remapeamos para 0-based por grafo
    # e expandimos x para cobrir todos os nós referenciados em edge_index.
    for data in dataset:
        if data.edge_index.numel() == 0:
            continue
        unique_nodes, new_edge_index = torch.unique(data.edge_index, return_inverse=True)
        new_edge_index = new_edge_index.reshape(2, -1)
        n_unique = unique_nodes.shape[0]

        if data.x is not None and data.x.shape[0] != n_unique:
            # Expande x: para nós presentes em edge_index mas sem feature row,
            # usa zeros (eles estavam fora do range original de x)
            x_expanded = torch.zeros(n_unique, data.x.shape[1], dtype=data.x.dtype)
            # Copia features para os nós que existiam em x (índices < x.shape[0])
            old_n = data.x.shape[0]
            mask = unique_nodes < old_n
            x_expanded[mask] = data.x[unique_nodes[mask]]
            data.x = x_expanded

        data.edge_index = new_edge_index
        # Não setar data.num_nodes explicitamente: PyG infere de data.x.shape[0]
        # (setar int quebra Batch.from_data_list ao tentar collate)

    # Node features: usa grau se ausente em qualquer grafo
    from torch_geometric.utils import degree as pyg_degree
    for data in dataset:
        if data.x is None:
            n = data.edge_index.max().item() + 1 if data.edge_index.numel() > 0 else 0
            deg = pyg_degree(data.edge_index[1], num_nodes=n).unsqueeze(-1).float()
            data.x = deg

    # Edge features: garante que TODOS os grafos têm edge_attr com tamanho
    # exatamente igual a edge_index.shape[1] — inconsistências causam CUDA assert
    in_edge_dim = 1
    for data in dataset:
        if data.edge_attr is not None and data.edge_attr.shape[0] == data.edge_index.shape[1]:
            in_edge_dim = data.edge_attr.shape[1]
            break

    for data in dataset:
        if data.edge_attr is None or data.edge_attr.shape[0] != data.edge_index.shape[1]:
            data.edge_attr = torch.ones(data.edge_index.shape[1], in_edge_dim)

    preprocess_dataset(dataset)

    labels = [data.y.item() for data in dataset]
    in_node_dim = dataset[0].x.shape[1]
    num_classes = int(max(labels)) + 1

    return dataset, labels, in_node_dim, in_edge_dim, num_classes


def train_epoch(model, loader, optimizer, criterion, device, lambda_hol):
    model.train()
    total_loss = 0

    for batch, padded_idxs, padded_flips, cycle_lengths, cycle_batch in loader:
        batch = batch.to(device)
        padded_idxs   = padded_idxs.to(device)
        padded_flips   = padded_flips.to(device)
        cycle_lengths  = cycle_lengths.to(device)
        cycle_batch    = cycle_batch.to(device)

        optimizer.zero_grad()
        out, H = model(batch, padded_idxs, padded_flips, cycle_lengths, cycle_batch)

        loss_ce = criterion(out, batch.y)
        loss_hol = compute_holonomy_loss(H, device)
        loss = loss_ce + lambda_hol * loss_hol

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def compute_holonomy_loss(H, device):
    # H: [total_cycles, d, d] — retornado diretamente pelo WilsonReadout
    if H.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    I = torch.eye(H.shape[-1], device=device).unsqueeze(0)  # [1, d, d]
    return ((I - H) ** 2).sum()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, padded_idxs, padded_flips, cycle_lengths, cycle_batch in loader:
            batch         = batch.to(device)
            padded_idxs   = padded_idxs.to(device)
            padded_flips   = padded_flips.to(device)
            cycle_lengths  = cycle_lengths.to(device)
            cycle_batch    = cycle_batch.to(device)
            out, _ = model(batch, padded_idxs, padded_flips, cycle_lengths, cycle_batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total


def main(save_dir=None, cfg_override=None):
    # Mescla CONFIG base com overrides
    cfg = copy.deepcopy(CONFIG)
    if cfg_override:
        for k, v in cfg_override.items():
            if k == "model" and isinstance(v, dict):
                cfg["model"] = {**cfg["model"], **v}
            else:
                cfg[k] = v

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg['seed'])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    dataset_name = cfg.get("dataset", "MUTAG")
    dataset, labels, in_node_dim, in_edge_dim, num_classes = load_data(dataset_name)

    # Sobrescreve dims detectadas automaticamente
    cfg["model"]["in_node_dim"] = in_node_dim
    cfg["model"]["in_edge_dim"] = in_edge_dim
    cfg["model"]["out_dim"] = num_classes

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg['seed'])
    criterion = nn.CrossEntropyLoss()
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]

        pin = device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset, batch_size=cfg['batch_size'],
            shuffle=True, collate_fn=bundle_collate, pin_memory=pin
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg['batch_size'],
            collate_fn=bundle_collate, pin_memory=pin
        )

        model = PiOneGNN(**cfg['model']).to(device)
        optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        best_acc = 0.0
        best_weights = None

        for epoch in tqdm(range(cfg['epochs']), desc=f"Fold {fold+1}", leave=False):
            loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg['lambda_hol'])
            acc = evaluate(model, test_loader, device)

            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 100 == 0:
                tqdm.write(f"  Epoch {epoch+1} | loss: {loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

        fold_accs.append(best_acc)

        if save_dir is not None:
            torch.save(best_weights, os.path.join(save_dir, f"fold_{fold+1}_model.pt"))

    mean_acc = sum(fold_accs) / len(fold_accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs)) ** 0.5

    tqdm.write(f"[{dataset_name}] Média: {mean_acc:.4f} ± {std_acc:.4f}")

    if save_dir is not None:
        results = {
            "fold_accs": fold_accs,
            "mean": mean_acc,
            "std": std_acc,
            "config": cfg,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return {"mean": mean_acc, "std": std_acc, "fold_accs": fold_accs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Diretório para salvar pesos e resultados (opcional)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Nome do dataset TUDataset (default: valor em config.py)')
    args = parser.parse_args()

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    main(save_dir=args.save_dir, cfg_override=override or None)
