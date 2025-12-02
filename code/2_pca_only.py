"""
Step B: Apply PCA to raw images (3072-D -> 200-D)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import time
import sys
import os
from config import *


# Simple Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, len(CLASS_NAMES))

    def forward(self, x):
        return self.fc(x)

def train_classifier(model, train_loader, test_loader, epochs):
    """Train and evaluate classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def main_pca():
    print("\n" + "="*60)
    print(f"STEP B: Applying PCA (3072-D → {PCA_COMPONENTS}-D)")
    if SAMPLE_SIZE:
        print(f"!!!!!!!!!!TESTING MODE: Using sample size = {SAMPLE_SIZE}")

    print("="*60 + "\n")
    
    # Load raw data
    try:
        data = np.load('cifar10-raw.npz')
        train_features = data['train_features']
        train_labels = data['train_labels']
        test_features = data['test_features']
        test_labels = data['test_labels']
        print(f"Loaded raw data: {train_features.shape}")
    except FileNotFoundError:
        print("Error: cifar10-raw.npz not found. Run 1_save_raw_images.py first.")
        sys.exit(1)
    
    # Apply PCA
    print(f"\nFitting PCA with {PCA_COMPONENTS} components...")
    start_time = time.time()
    
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(train_features)
    
    train_pca = pca.transform(train_features)
    test_pca = pca.transform(test_features)
    
    pca_time = time.time() - start_time
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA completed in {pca_time:.2f} seconds")
    print(f"Variance explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
    print(f"Reduced shape: {train_pca.shape}")
    
    # Calculate reconstruction error
    train_reconstructed = pca.inverse_transform(train_pca)
    mse_loss = np.mean((train_features - train_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse_loss:.6f}")
    
    # Save PCA features for autoencoder/transformer
    os.makedirs('../data', exist_ok=True)
    print(f"\nSaving PCA features to '../data/cifar10-pca{PCA_COMPONENTS}.npz'...")
    np.savez_compressed(
        f'../data/cifar10-pca{PCA_COMPONENTS}',
        train_features=train_pca,
        train_labels=train_labels,
        test_features=test_pca,
        test_labels=test_labels
    )
    
    # Train classifier on PCA features
    print(f"\nTraining classifier on PCA-{PCA_COMPONENTS} features...")
    train_pca_tensor = torch.from_numpy(train_pca).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    test_pca_tensor = torch.from_numpy(test_pca).float()
    test_labels_tensor = torch.from_numpy(test_labels).long()
    
    train_dataset = TensorDataset(train_pca_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_pca_tensor, test_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    classifier = Classifier(PCA_COMPONENTS)
    accuracy = train_classifier(classifier, train_loader, test_loader, EPOCHS)
    
    print(f"\n{'='*60}")
    print(f"PCA-{PCA_COMPONENTS} CLASSIFICATION ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs('../results', exist_ok=True)
    with open('../results/pca_results.txt', 'w') as f:
        f.write(f"PCA Results\n")
        f.write(f"="*40 + "\n")
        f.write(f"Components: {PCA_COMPONENTS}\n")
        f.write(f"Variance Explained: {variance_explained:.4f}\n")
        f.write(f"Reconstruction MSE: {mse_loss:.6f}\n")
        f.write(f"Classification Accuracy: {accuracy:.2f}%\n")
        f.write(f"PCA Time: {pca_time:.2f} seconds\n")
    
    # Also save to CSV
    import csv
    pca_csv = '../results/pca_results.csv'
    with open(pca_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Input_Dim', 'Output_Dim', 'Compression_Ratio',
                        'Variance_Explained', 'MSE_Loss', 'Accuracy_%', 'Training_Time_sec'])
        writer.writerow(['PCA', 3072, PCA_COMPONENTS, 
                        f'{3072/PCA_COMPONENTS:.1f}:1',
                        variance_explained, mse_loss, accuracy, pca_time])
    

    return accuracy, mse_loss, pca_time

def main():
    main_pca()

if __name__ == "__main__":
    main()