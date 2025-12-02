"""
Autoencoder Classifier for CIFAR-100
Can work on:
1. Raw images (3072-D)
2. PCA features (512-D)

Note: This is an autoencoder used as a CLASSIFIER, not for dimensionality reduction.
It learns features through reconstruction, then classifies.
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


class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AutoencoderClassifier, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (for reconstruction loss)
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classifier head (from bottleneck)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1] // 2, NUM_CLASSES)
        )
    
    def forward(self, x, return_reconstruction=False):
        # Encode
        encoded = self.encoder(x)
        
        # Classify
        classification = self.classifier(encoded)
        
        if return_reconstruction:
            # Decode (for reconstruction loss during training)
            reconstructed = self.decoder(encoded)
            return classification, reconstructed
        else:
            return classification


def train_autoencoder_classifier(model, train_loader, test_loader, device, epochs):
    """Train autoencoder classifier with joint classification and reconstruction loss"""
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss weights: balance classification and reconstruction
    alpha = 0.7  # Classification weight
    beta = 0.3   # Reconstruction weight
    
    print("\nTraining Autoencoder Classifier...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_class_loss = 0.0
        running_recon_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with reconstruction
            class_outputs, reconstructed = model(inputs, return_reconstruction=True)
            
            # Combined loss
            class_loss = classification_criterion(class_outputs, labels)
            recon_loss = reconstruction_criterion(reconstructed, inputs)
            loss = alpha * class_loss + beta * recon_loss
            
            loss.backward()
            optimizer.step()
            
            running_class_loss += class_loss.item()
            running_recon_loss += recon_loss.item()
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_loader)}], "
                      f"Class Loss: {class_loss.item():.4f}, Recon Loss: {recon_loss.item():.6f}")
        
        epoch_class_loss = running_class_loss / len(train_loader)
        epoch_recon_loss = running_recon_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} - Class Loss: {epoch_class_loss:.4f}, "
              f"Recon Loss: {epoch_recon_loss:.6f}, Train Acc: {epoch_acc:.2f}%")
    
    train_time = time.time() - start_time
    
    # Evaluate (classification only)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, return_reconstruction=False)
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
    
    # Test 1: Autoencoder on raw images
    print("\n" + "="*60)
    print("TEST 1: Autoencoder on Raw Images (3072-D)")
    print("="*60)
    
    data = np.load('../data/cifar100_raw.npz')
    train_features = torch.from_numpy(data['train_features']).float()
    train_labels = torch.from_numpy(data['train_labels']).long()
    test_features = torch.from_numpy(data['test_features']).float()
    test_labels = torch.from_numpy(data['test_labels']).long()
    
    # Hidden dims for raw images
    hidden_dims = [1024, 512, 256]
    model = AutoencoderClassifier(FLATTENED_DIM, hidden_dims).to(device)
    
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    acc, time_taken = train_autoencoder_classifier(model, train_loader, test_loader, device, EPOCHS)
    print(f"\n✅ Autoencoder (Raw) - Accuracy: {acc:.2f}%, Time: {time_taken:.2f}s")
    results.append(['Autoencoder', 'Raw', FLATTENED_DIM, acc, time_taken])
    
    # Test 2: Autoencoder on PCA features
    print("\n" + "="*60)
    print(f"TEST 2: Autoencoder on PCA Features ({PCA_COMPONENTS}-D)")
    print("="*60)
    
    data = np.load(f'../data/cifar100_pca{PCA_COMPONENTS}.npz')
    train_features = torch.from_numpy(data['train_features']).float()
    train_labels = torch.from_numpy(data['train_labels']).long()
    test_features = torch.from_numpy(data['test_features']).float()
    test_labels = torch.from_numpy(data['test_labels']).long()
    
    # Hidden dims for PCA features
    hidden_dims = [256, 128, 64]
    model = AutoencoderClassifier(PCA_COMPONENTS, hidden_dims).to(device)
    
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    acc, time_taken = train_autoencoder_classifier(model, train_loader, test_loader, device, EPOCHS)
    print(f"\n✅ Autoencoder (PCA) - Accuracy: {acc:.2f}%, Time: {time_taken:.2f}s")
    results.append(['Autoencoder', f'PCA-{PCA_COMPONENTS}', PCA_COMPONENTS, acc, time_taken])
    
    # Save results
    with open('../results/autoencoder_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Input_Type', 'Input_Dim', 'Accuracy_%', 'Training_Time_sec'])
        writer.writerows(results)
    
    print(f"\n✅ Results saved to '../results/autoencoder_results.csv'\n")


if __name__ == "__main__":
    main()