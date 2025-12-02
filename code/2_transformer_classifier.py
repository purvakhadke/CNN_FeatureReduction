"""
Transformer Classifier for CIFAR-100
Can work on:
1. Raw images (3072-D as sequence of patches)
2. PCA features (512-D as sequence)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import csv
import os
from config import *


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, patch_size, num_patches):
        super(TransformerClassifier, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_size, TRANSFORMER_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, TRANSFORMER_DIM))
        self.cls_token = nn.Parameter(torch.randn(1, 1, TRANSFORMER_DIM))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_DIM * 4,
            dropout=TRANSFORMER_DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(TRANSFORMER_DIM),
            nn.Linear(TRANSFORMER_DIM, NUM_CLASSES)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create patches: (batch, input_dim) -> (batch, num_patches, patch_size)
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Embed patches
        x = self.patch_embedding(x)  # (batch, num_patches, transformer_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, transformer_dim)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        
        return output


def train_transformer(model, train_loader, test_loader, device, epochs):
    """Train transformer classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    print("\nTraining Transformer...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, train_time


def main():
    os.makedirs('../results', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = []
    
    # Test 1: Transformer on raw images
    print("\n" + "="*60)
    print("TEST 1: Transformer on Raw Images (3072-D)")
    print("="*60)
    
    data = np.load('../data/cifar100_raw.npz')
    train_features = torch.from_numpy(data['train_features']).float()
    train_labels = torch.from_numpy(data['train_labels']).long()
    test_features = torch.from_numpy(data['test_features']).float()
    test_labels = torch.from_numpy(data['test_labels']).long()
    
    # Patch configuration for 3072-D: 64 patches of 48-D each
    num_patches = 64
    patch_size = FLATTENED_DIM // num_patches
    
    model = TransformerClassifier(FLATTENED_DIM, patch_size, num_patches).to(device)
    
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    acc, time_taken = train_transformer(model, train_loader, test_loader, device, EPOCHS)
    print(f"\n✅ Transformer (Raw) - Accuracy: {acc:.2f}%, Time: {time_taken:.2f}s")
    results.append(['Transformer', 'Raw', FLATTENED_DIM, acc, time_taken])
    
    # Test 2: Transformer on PCA features
    print("\n" + "="*60)
    print(f"TEST 2: Transformer on PCA Features ({PCA_COMPONENTS}-D)")
    print("="*60)
    
    data = np.load(f'../data/cifar100_pca{PCA_COMPONENTS}.npz')
    train_features = torch.from_numpy(data['train_features']).float()
    train_labels = torch.from_numpy(data['train_labels']).long()
    test_features = torch.from_numpy(data['test_features']).float()
    test_labels = torch.from_numpy(data['test_labels']).long()
    
    # Patch configuration for PCA: 32 patches
    num_patches = 32
    patch_size = PCA_COMPONENTS // num_patches
    
    model = TransformerClassifier(PCA_COMPONENTS, patch_size, num_patches).to(device)
    
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    acc, time_taken = train_transformer(model, train_loader, test_loader, device, EPOCHS)
    print(f"\n✅ Transformer (PCA) - Accuracy: {acc:.2f}%, Time: {time_taken:.2f}s")
    results.append(['Transformer', f'PCA-{PCA_COMPONENTS}', PCA_COMPONENTS, acc, time_taken])
    
    # Save results
    with open('../results/transformer_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Input_Type', 'Input_Dim', 'Accuracy_%', 'Training_Time_sec'])
        writer.writerows(results)
    
    print(f"\n✅ Results saved to '../results/transformer_results.csv'\n")


if __name__ == "__main__":
    main()