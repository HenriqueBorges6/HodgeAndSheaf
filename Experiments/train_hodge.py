"""
Training script for HodgeGNN on MUTAG dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.hodge_gnn import HodgeGNN
from utils.data_loader import load_mutag_dataset, MUTAGDataset
from utils.hodge_utils import compute_hodge_laplacian_L1, prepare_edge_features


def collate_fn(batch):
    """
    Custom collate function for batching graphs

    Returns list of graphs instead of stacking
    (graphs have different sizes)
    """
    return batch


def train_epoch(model, dataloader, optimizer, criterion, device, num_edge_types):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_preds = []
        batch_labels = []

        for sample in batch:
            graph_data = sample['graph_data']
            label = sample['label']

            # Prepare data
            edges = graph_data['edges']
            edge_labels = graph_data['edge_features']
            num_nodes = graph_data['num_nodes']

            # Compute Hodge Laplacian L1
            L1 = compute_hodge_laplacian_L1(edges, num_nodes)
            L1 = torch.tensor(L1, dtype=torch.float32).to(device)

            # Prepare edge features (one-hot encoding)
            edge_features = prepare_edge_features(edge_labels, num_edge_types).to(device)

            # Forward pass
            logits = model(L1, edge_features)

            # Loss
            target = torch.tensor(label, dtype=torch.long).to(device)
            loss = criterion(logits.unsqueeze(0), target.unsqueeze(0))
            batch_loss += loss

            # Predictions
            pred = torch.argmax(logits).item()
            batch_preds.append(pred)
            batch_labels.append(label)

        # Backward pass
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

    # Metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, num_edge_types):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch_loss = 0.0
            batch_preds = []
            batch_labels = []

            for sample in batch:
                graph_data = sample['graph_data']
                label = sample['label']

                # Prepare data
                edges = graph_data['edges']
                edge_labels = graph_data['edge_features']
                num_nodes = graph_data['num_nodes']

                # Compute Hodge Laplacian L1
                L1 = compute_hodge_laplacian_L1(edges, num_nodes)
                L1 = torch.tensor(L1, dtype=torch.float32).to(device)

                # Prepare edge features
                edge_features = prepare_edge_features(edge_labels, num_edge_types).to(device)

                # Forward pass
                logits = model(L1, edge_features)

                # Loss
                target = torch.tensor(label, dtype=torch.long).to(device)
                loss = criterion(logits.unsqueeze(0), target.unsqueeze(0))
                batch_loss += loss

                # Predictions
                pred = torch.argmax(logits).item()
                batch_preds.append(pred)
                batch_labels.append(label)

            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss.item()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    # Metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main():
    # Hyperparameters
    hidden_dims = [32, 32, 32, 32]
    num_classes = 2
    mlp_hidden_dim = 512
    pooling_type = 'sum'
    dropout = 0.5

    batch_size = 8
    num_epochs = 30
    learning_rate = 0.01  # Reduced from 0.1 - critical fix!
    weight_decay = 5e-4
    train_ratio = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("\n=== Loading MUTAG Dataset ===")
    mutag_data = load_mutag_dataset()
    dataset = MUTAGDataset(mutag_data)

    # Detect number of edge types from dataset
    edge_feature_dim = len(np.unique(mutag_data['edge_labels']))
    print(f"Detected {edge_feature_dim} edge types in MUTAG dataset")

    # Train/test split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Train size: {train_size}, Test size: {test_size}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    print("\n=== Initializing HodgeGNN ===")
    model = HodgeGNN(
        edge_feature_dim=edge_feature_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        mlp_hidden_dim=mlp_hidden_dim,
        pooling_type=pooling_type,
        dropout=dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n=== Training HodgeGNN ===")
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, edge_feature_dim)
        test_loss, test_acc, test_prec, test_rec, test_f1, _, _ = evaluate(model, test_loader, criterion, device, edge_feature_dim)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), 'best_hodge_gnn.pt')

    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load('best_hodge_gnn.pt'))
    test_loss, test_acc, test_prec, test_rec, test_f1, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device, edge_feature_dim
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
