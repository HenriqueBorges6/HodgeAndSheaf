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


def load_data():
    raw_dataset = TUDataset(root='data/MUTAG', name='MUTAG')
    dataset = list(raw_dataset)  # materializa todos os objetos em memória
    preprocess_dataset(dataset)  # modifica os objetos in-place
    labels = [data.y.item() for data in dataset]
    return dataset, labels


def train_epoch(model, loader, optimizer, criterion, device, lambda_hol):
    model.train()
    total_loss = 0

    for batch, cycles, edge_map, cycle_batch in loader:
        batch = batch.to(device)
        cycle_batch = cycle_batch.to(device)

        optimizer.zero_grad()
        out, holonomies = model(batch, cycles, edge_map, cycle_batch)

        loss_ce = criterion(out, batch.y)
        loss_hol = compute_holonomy_loss(holonomies, device)
        loss = loss_ce + lambda_hol * loss_hol

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def compute_holonomy_loss(holonomies, device):
    d = holonomies[0].shape[-1]
    I = torch.eye(d, device=device)
    loss = sum(torch.norm(I - H, p='fro') ** 2 for H in holonomies)
    return loss


# def compute_holonomy_loss(O_edges, cycles, edge_map, device):
#     loss = torch.tensor(0.0, device=device)
#     d = O_edges.shape[-1]
#     I = torch.eye(d, device=device)

#     for cycle in cycles:
#         H = torch.eye(d, device=device)
#         for (src, dst) in cycle:
#             idx, is_reversed = edge_map[(src, dst)]
#             O = O_edges[idx]
#             if is_reversed:
#                 O = O.transpose(-1, -2)
#             H = O @ H
#         loss = loss + torch.norm(I - H, p='fro') ** 2

#     return loss


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, cycles, edge_map, cycle_batch in loader:
            batch = batch.to(device)
            cycle_batch = cycle_batch.to(device)
            out, _ = model(batch, cycles, edge_map, cycle_batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(CONFIG['seed'])

    dataset, labels = load_data()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=CONFIG['seed'])
    criterion = nn.CrossEntropyLoss()
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]

        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG['batch_size'],
            shuffle=True, collate_fn=bundle_collate
        )
        test_loader = DataLoader(
            test_dataset, batch_size=CONFIG['batch_size'],
            collate_fn=bundle_collate
        )

        model = PiOneGNN(**CONFIG['model']).to(device)
        optimizer = Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

        for epoch in tqdm(range(CONFIG['epochs']), desc=f"Fold {fold+1}"):
            loss = train_epoch(model, train_loader, optimizer, criterion, device, CONFIG['lambda_hol'])
            acc = evaluate(model, test_loader, device)
            if (epoch + 1) % 20 == 0:
                tqdm.write(f"Epoch {epoch+1} | loss: {loss:.4f} | Acc: {acc:.4f}")

        fold_accs.append(acc)

    print(f"Média: {sum(fold_accs)/len(fold_accs):.4f}")


if __name__ == '__main__':
    main()
