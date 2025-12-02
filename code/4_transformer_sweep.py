"""
Step D: Apply Transformer to PCA features (200-D -> [10, 25, 50, 100])
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

# Configuration
PCA_INPUT_DIM = 200
DIMENSIONS_TO_COMPRESS_TO = [10, 25, 50, 100]
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30
EPOCHS_CLASSIFIER = 10
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Transformer configuration for 200-D input
PATCH_COUNT = 10  # Split 200-D into 10 patches
PATCH_SIZE = 20   # Each patch is 20-D
TRANSFORMER_DIM = 64

# Transformer-based Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TransformerAutoencoder, self).__init__()
        
        # Patch embedding
        self.patch_embedding = nn.Linear(PATCH_SIZE, TRANSFORMER_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, PATCH_COUNT, TRANSFORMER_DIM))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=4,
            dim_feedforward=TRANSFORMER_DIM * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Compression
        self.compressor = nn.Linear(PATCH_COUNT * TRANSFORMER_DIM, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create patches
        x_seq = x.view(batch_size, PATCH_COUNT, PATCH_SIZE)
        
        # Embed and add position
        x_emb = self.patch_embedding(x_seq)
        x_emb = x_emb + self.pos_embedding
        
        # Transformer
        trans_out = self.transformer_encoder(x_emb)
        
        # Compress
        flat = trans_out.reshape(batch_size, -1)
        latent = self.compressor(flat)
        
        # Reconstruct
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent

# Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, len(CLASSES))
        )

    def forward(self, x):
        return self.net(x)

def train_transformer(model, loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
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
            
    accuracy = 100 * correct / total
    return accuracy

def main_sweep():
    print("\n" + "="*60)
    print(f"STEP D: Transformer Sweep (PCA-{PCA_INPUT_DIM} → Compressed)")
    print("="*60 + "\n")
    
    # Load PCA features
    try:
        data = np.load(f'../data/cifar10-pca{PCA_INPUT_DIM}.npz')
        train_features = torch.from_numpy(data['train_features']).float()
        train_labels = torch.from_numpy(data['train_labels']).long()
        test_features = torch.from_numpy(data['test_features']).float()
        test_labels = torch.from_numpy(data['test_labels']).long()
        print(f"Loaded PCA-{PCA_INPUT_DIM} features: {train_features.shape}")
    except FileNotFoundError:
        print(f"Error: cifar10-pca{PCA_INPUT_DIM}.npz not found. Run 2_pca_only.py first.")
        sys.exit(1)

    train_data = TensorDataset(train_features, train_labels)

    results_mse = []
    results_acc = []
    results_time = []

    for latent_dim in DIMENSIONS_TO_COMPRESS_TO:
        print(f"\n--- Transformer: {PCA_INPUT_DIM}D → {latent_dim}D ---")
        
        model_path = f'../models/transformer_pca{PCA_INPUT_DIM}_to_{latent_dim}d.pth'
        os.makedirs('../models/', exist_ok=True)
        
        model = TransformerAutoencoder(PCA_INPUT_DIM, latent_dim)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state'])
            final_mse = checkpoint['final_mse_loss']
            train_time = checkpoint['train_time']
            print(f"Loaded pre-trained model")
        else:
            start_time = time.time()
            final_mse = train_transformer(model, train_loader, EPOCHS)
            train_time = time.time() - start_time
            torch.save({
                'model_state': model.state_dict(),
                'final_mse_loss': final_mse,
                'train_time': train_time
            }, model_path)
            print(f"Training completed in {train_time:.2f}s")
            
        results_mse.append(final_mse)
        results_time.append(train_time)
        
        # Get compressed features
        model.eval()
        with torch.no_grad():
            _, train_compressed = model(train_features)
            _, test_compressed = model(test_features)
            
        # Train classifier
        cls_train_ds = TensorDataset(train_compressed, train_labels)
        cls_test_ds = TensorDataset(test_compressed, test_labels)
        cls_train_loader = DataLoader(cls_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        classifier = Classifier(latent_dim)
        accuracy = train_classifier(classifier, cls_train_loader, cls_test_loader, EPOCHS_CLASSIFIER)
        results_acc.append(accuracy)
        
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Classifier Accuracy: {accuracy:.2f}%")

    # Save results
    os.makedirs('../results/', exist_ok=True)
    with open('../results/transformer_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Latent_Dim', 'MSE_Loss', 'Accuracy', 'Training_Time_sec'])
        for dim, mse, acc, t in zip(DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, results_time):
            writer.writerow([dim, mse, acc, t])
    
    print(f"\n✅ Results saved to '../results/transformer_results.csv'\n")

def main():
    main_sweep()

if __name__ == "__main__":
    main()