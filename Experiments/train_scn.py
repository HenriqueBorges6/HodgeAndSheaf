"""
Training script for Sheaf Convolutional Network (SCN) on MUTAG dataset

Based on "Neural Sheaf Diffusion" (Bodnar et al., NeurIPS 2022)
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

from models.sheaf_convolutional_network import SheafConvolutionalNetwork
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

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    print("=" * 70)
    print("Sheaf Convolutional Network (SCN) - MUTAG Classification")
    print("Based on: Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022)")
    print("=" * 70)

    # Hyperparameters
    # Experiment with different sheaf types: 'diagonal', 'orthogonal', 'general'
    sheaf_type = 'diagonal'  # Start with diagonal (simplest)
    stalk_dim = 3  # d in the paper
    hidden_dims = [32, 32, 32]  # Multiple SCN layers
    num_classes = 2
    mlp_hidden_dim = 128
    dropout = 0.5

    batch_size = 8
    num_epochs = 10
    learning_rate = 0.01
    weight_decay = 5e-3
    train_ratio = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Sheaf type: {sheaf_type}")
    print(f"Stalk dimension: {stalk_dim}")
    print(f"Hidden dimensions: {hidden_dims}")

    # Load dataset
    print("\n" + "=" * 70)
    print("Loading MUTAG Dataset")
    print("=" * 70)
    mutag_data = load_mutag_dataset()
    dataset = MUTAGDataset(mutag_data)

    # Detect number of edge types
    edge_feature_dim = len(np.unique(mutag_data['edge_labels']))
    print(f"\nEdge feature dimension: {edge_feature_dim}")

    # Train/test split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    print("\n" + "=" * 70)
    print("Initializing Sheaf Convolutional Network")
    print("=" * 70)
    model = SheafConvolutionalNetwork(
        edge_feature_dim=edge_feature_dim,
        hidden_dims=hidden_dims,
        stalk_dim=stalk_dim,
        num_classes=num_classes,
        sheaf_type=sheaf_type,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout
    ).to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, # verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, edge_feature_dim)
        test_loss, test_acc, test_prec, test_rec, test_f1, _, _ = evaluate(
            model, test_loader, criterion, device, edge_feature_dim
        )

        # Update learning rate
        scheduler.step(test_acc)

        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"Test F1: {test_f1:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), 'best_scn_model.pt')
            print(f"  ✓ New best model saved (acc: {best_test_acc:.4f})")

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    model.load_state_dict(torch.load('best_scn_model.pt'))
    test_loss, test_acc, test_prec, test_rec, test_f1, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device, edge_feature_dim
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nClass 0 (non-mutagenic): {cm[0, 0]} correct, {cm[0, 1]} incorrect")
    print(f"Class 1 (mutagenic):     {cm[1, 1]} correct, {cm[1, 0]} incorrect")

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
