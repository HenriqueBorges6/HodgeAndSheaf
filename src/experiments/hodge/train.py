import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from src.experiments.hodge.config import CONFIG
from src.models.Hodge import HodgeGNN

from tqdm import tqdm

def load_data():
    dataset = TUDataset(root='data/MUTAG', name = 'MUTAG')
    labels = [data.y.item() for data in dataset]
    return dataset, labels

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return (total_loss / len(loader))

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(CONFIG['seed'])
    dataset, labels  = load_data()
    skf = StratifiedKFold(n_splits= 10, shuffle=True, random_state=CONFIG['seed'])
    criterion = nn.CrossEntropyLoss()
    fold_accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset,labels)):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

        model = HodgeGNN(**CONFIG['model']).to(device)
        optimizer = Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        for epoch in tqdm(range(CONFIG['epochs']), desc=f"Fold {fold+1}"):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            acc = evaluate(model, test_loader, device)
            if (epoch + 1) % 20 == 0:
                tqdm.write(f"Epoch {epoch+1} | loss: {loss: .4f} | Acc: {acc: .4f}")
        fold_accs.append(acc)
    
    print(f"Média: {sum(fold_accs)/len(fold_accs):.4f}")

if __name__ == '__main__':
    main()
