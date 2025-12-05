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
        # apparently GeLU is better than ReLU for transformer image classigncaion
        # lets mess around with this to see what different activations do (if we have time)
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
        
        # Create and embed patches
        x = x.view(batch_size, self.num_patches, self.patch_size)
        x = self.patch_embedding(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        
        return output


def get_patch_config(input_dim):
    if input_dim == (32*32*3):
        num_patches = 64
    
    # else its 512, the reduced dimensions (512)
    else:
        num_patches = 32
    
    patch_size = input_dim // num_patches
    return num_patches, patch_size


def train_transformer(model, train_loader, test_loader, device, epochs, input_type):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
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
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Train Acc: {epoch_acc:.2f}%")
    
    train_time = time.time() - start_time
    
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
    os.makedirs('../models', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    for input_type, data_path in INPUT_FILES.items():
        # Load the data
        data = np.load(data_path)
        train_features = torch.from_numpy(data['train_features']).float()
        train_labels = torch.from_numpy(data['train_labels']).long()
        test_features = torch.from_numpy(data['test_features']).float()
        test_labels = torch.from_numpy(data['test_labels']).long()
        
        # if its raw then input is 32x32x3 else its the reduced 512
        is_raw = (input_type == 'Raw')
        input_dim = 32*32*3 if is_raw else REDUCED_DIM
        
        # Get patch configs
        num_patches, patch_size = get_patch_config(input_dim)
        
        # Build model
        model = TransformerClassifier(input_dim, patch_size, num_patches).to(device)
        
        # Create dataloaders
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train
        acc, time_taken = train_transformer(model, train_loader, test_loader, device, EPOCHS, input_type)
        
        # Save model
        model_name = f"transformer_{input_type.lower().replace('-', '_')}.pth"
        torch.save(model.state_dict(), f'../models/{model_name}')
        
        results.append(['Transformer', input_type, input_dim, acc, time_taken])
    
    # Save results into CSV for the reporr
    with open('../results/transformer_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Input_Type', 'Input_Dim', 'Accuracy_%', 'Training_Time_sec'])
        writer.writerows(results)

if __name__ == "__main__":
    main()