"""
Step C: Apply Autoencoder to PCA features (200-D -> [10, 25, 50, 100])
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os
import time
import csv
from config import *


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Classifier Model
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, NUM_CLASSES)

    def forward(self, x):
        return self.fc(x)

def train_autoencoder(model, loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in loader:
            inputs = data[0]
            optimizer.zero_grad()
            reconstruction, _ = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.6f}")
    
    return epoch_loss / len(loader)

def train_classifier(model, train_loader, test_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = (correct / total) * 100
    return accuracy

def main_sweep():
    print("\n" + "="*60)
    print(f"STEP C: Autoencoder Sweep (PCA-{PCA_COMPONENTS} → Compressed)")
    print("="*60 + "\n")
    
    # Load PCA features
    try:
        data = np.load(f'../data/cifar100-pca{PCA_COMPONENTS}.npz')
        train_features = torch.from_numpy(data['train_features']).float()
        train_labels = torch.from_numpy(data['train_labels']).long()
        test_features = torch.from_numpy(data['test_features']).float()
        test_labels = torch.from_numpy(data['test_labels']).long()
        print(f"Loaded PCA-{PCA_COMPONENTS} features: {train_features.shape}")
    except FileNotFoundError:
        print(f"Error: cifar100-pca{PCA_COMPONENTS}.npz not found. Run 2_pca_only.py first.")
        sys.exit(1)

    train_data = TensorDataset(train_features, train_labels)

    results_mse = []
    results_acc = []
    results_time = []

    for latent_dim in DIMENSIONS_TO_COMPRESS_TO:
        print(f"\n--- Autoencoder: {PCA_COMPONENTS}D → {latent_dim}D ---")
        
        model_path = f'../models/autoencoder_pca{PCA_COMPONENTS}_to_{latent_dim}d.pth'
        os.makedirs('../models/', exist_ok=True)
        
        autoencoder = Autoencoder(PCA_COMPONENTS, latent_dim)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            autoencoder.load_state_dict(checkpoint['model_state'])
            final_mse = checkpoint['final_mse_loss']
            train_time = checkpoint['train_time']
            print(f"Loaded pre-trained model")
        else:
            ae_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
            start_time = time.time()
            final_mse = train_autoencoder(autoencoder, ae_loader, EPOCHS)
            train_time = time.time() - start_time
            torch.save({
                'model_state': autoencoder.state_dict(),
                'final_mse_loss': final_mse,
                'train_time': train_time
            }, model_path)
            print(f"Training completed in {train_time:.2f}s")

        results_mse.append(final_mse)
        results_time.append(train_time)

        # Get compressed features
        autoencoder.eval()
        with torch.no_grad():
            train_compressed = autoencoder.encoder(train_features)
            test_compressed = autoencoder.encoder(test_features)

        # Train classifier
        cls_train_dataset = TensorDataset(train_compressed, train_labels)
        cls_test_dataset = TensorDataset(test_compressed, test_labels)
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        classifier = Classifier(latent_dim)
        accuracy = train_classifier(classifier, cls_train_loader, cls_test_loader, NUM_CLASSES)
        results_acc.append(accuracy)
        
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Classifier Accuracy: {accuracy:.2f}%")

    # Save results
    os.makedirs('../results/', exist_ok=True)
    with open('../results/autoencoder_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Latent_Dim', 'MSE_Loss', 'Accuracy', 'Training_Time_sec'])
        for dim, mse, acc, t in zip(DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, results_time):
            writer.writerow([dim, mse, acc, t])
    
    print(f"\n✅ Results saved to '../results/autoencoder_results.csv'\n")

def main():
    main_sweep()

if __name__ == "__main__":
    main()