"""
Step 0: Prepare CIFAR-100 data with multiple dimensionality reduction methods
- Raw (flattened)
- PCA (linear)
- UMAP (nonlinear, preserves local structure)
- Autoencoder (nonlinear, learned features)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from umap import UMAP
from config import *


# ============================================================
# Autoencoder for Dimensionality Reduction
# ============================================================
class DimReductionAutoencoder(nn.Module):
    """Autoencoder specifically for dimensionality reduction (not classification)"""
    def __init__(self, input_dim, latent_dim):
        super(DimReductionAutoencoder, self).__init__()
        
        # Encoder: input_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Decoder: latent_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder_reducer(train_data, latent_dim, device, epochs=30, batch_size=128):
    """Train autoencoder for dimensionality reduction"""
    print(f"\n  Training Autoencoder for dim reduction ({FLATTENED_DIM}-D → {latent_dim}-D)...")
    
    model = DimReductionAutoencoder(FLATTENED_DIM, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloader
    train_tensor = torch.from_numpy(train_data).float()
    dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target for autoencoder
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {avg_loss:.6f}")
    
    return model


def transform_with_autoencoder(model, data, device, batch_size=256):
    """Transform data using trained autoencoder encoder"""
    model.eval()
    data_tensor = torch.from_numpy(data).float()
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    encoded_list = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            encoded = model.encode(batch_x)
            encoded_list.append(encoded.cpu().numpy())
    
    return np.concatenate(encoded_list, axis=0)


def save_data():
    print("\n" + "="*60)
    print("Step 0: Preparing CIFAR-100 Data")
    print("="*60 + "\n")
    
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models/reducers', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ============================================================
    # Load CIFAR-100
    # ============================================================
    print("\n[1/5] Loading CIFAR-100...")
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True)
    
    # Flatten images: (N, 32, 32, 3) -> (N, 3072)
    train_images = trainset.data.reshape(len(trainset.data), -1).astype(np.float32) / 255.0
    train_labels = np.array(trainset.targets)
    test_images = testset.data.reshape(len(testset.data), -1).astype(np.float32) / 255.0
    test_labels = np.array(testset.targets)
    
    # Apply sample size if testing
    if TRAIN_SAMPLE_SIZE is not None:
        print(f"  Using sample size: Train={TRAIN_SAMPLE_SIZE}, Test={TEST_SAMPLE_SIZE}")
        train_images = train_images[:TRAIN_SAMPLE_SIZE]
        train_labels = train_labels[:TRAIN_SAMPLE_SIZE]
        test_images = test_images[:TEST_SAMPLE_SIZE]
        test_labels = test_labels[:TEST_SAMPLE_SIZE]
    
    print(f"  Train shape: {train_images.shape}, Test shape: {test_images.shape}")
    
    # ============================================================
    # Save Raw Data
    # ============================================================
    print("\n[2/5] Saving raw data (3072-D)...")
    np.savez_compressed(
        '../data/cifar100_raw.npz',
        train_features=train_images,
        train_labels=train_labels,
        test_features=test_images,
        test_labels=test_labels
    )
    print("  ✅ Saved to '../data/cifar100_raw.npz'")
    
    # ============================================================
    # PCA Reduction
    # ============================================================
    print(f"\n[3/5] Applying PCA ({FLATTENED_DIM}-D → {REDUCED_DIM}-D)...")
    pca = PCA(n_components=REDUCED_DIM, random_state=RANDOM_SEED)
    pca.fit(train_images)
    
    train_pca = pca.transform(train_images)
    test_pca = pca.transform(test_images)
    
    pca_variance = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {pca_variance:.4f} ({pca_variance*100:.2f}%)")
    
    # Save PCA model and data
    joblib.dump(pca, '../models/reducers/pca_model.joblib')
    np.savez_compressed(
        f'../data/cifar100_pca{REDUCED_DIM}.npz',
        train_features=train_pca,
        train_labels=train_labels,
        test_features=test_pca,
        test_labels=test_labels,
        variance_explained=pca_variance
    )
    print(f"  ✅ Saved to '../data/cifar100_pca{REDUCED_DIM}.npz'")
    print(f"  ✅ PCA model saved to '../models/reducers/pca_model.joblib'")
    
    # ============================================================
    # UMAP Reduction
    # ============================================================
    print(f"\n[4/5] Applying UMAP ({FLATTENED_DIM}-D → {REDUCED_DIM}-D)...")
    print("  (This may take a few minutes...)")
    
    umap_reducer = UMAP(
        n_components=REDUCED_DIM,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='euclidean',
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    # Fit on training data
    train_umap = umap_reducer.fit_transform(train_images)
    # Transform test data using fitted model
    test_umap = umap_reducer.transform(test_images)
    
    # Save UMAP model and data
    joblib.dump(umap_reducer, '../models/reducers/umap_model.joblib')
    np.savez_compressed(
        f'../data/cifar100_umap{REDUCED_DIM}.npz',
        train_features=train_umap,
        train_labels=train_labels,
        test_features=test_umap,
        test_labels=test_labels
    )
    print(f"  ✅ Saved to '../data/cifar100_umap{REDUCED_DIM}.npz'")
    print(f"  ✅ UMAP model saved to '../models/reducers/umap_model.joblib'")
    
    # ============================================================
    # Autoencoder Reduction
    # ============================================================
    print(f"\n[5/5] Applying Autoencoder ({FLATTENED_DIM}-D → {REDUCED_DIM}-D)...")
    
    ae_model = train_autoencoder_reducer(
        train_images, 
        latent_dim=REDUCED_DIM, 
        device=device, 
        epochs=AE_REDUCER_EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Transform data
    train_ae = transform_with_autoencoder(ae_model, train_images, device)
    test_ae = transform_with_autoencoder(ae_model, test_images, device)
    
    # Save Autoencoder model and data
    torch.save(ae_model.state_dict(), '../models/reducers/autoencoder_reducer.pth')
    np.savez_compressed(
        f'../data/cifar100_ae{REDUCED_DIM}.npz',
        train_features=train_ae,
        train_labels=train_labels,
        test_features=test_ae,
        test_labels=test_labels
    )
    print(f"  ✅ Saved to '../data/cifar100_ae{REDUCED_DIM}.npz'")
    print(f"  ✅ Autoencoder model saved to '../models/reducers/autoencoder_reducer.pth'")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print(f"  • cifar100_raw.npz         - Raw features ({FLATTENED_DIM}-D)")
    print(f"  • cifar100_pca{REDUCED_DIM}.npz       - PCA features ({REDUCED_DIM}-D, {pca_variance*100:.1f}% var)")
    print(f"  • cifar100_umap{REDUCED_DIM}.npz      - UMAP features ({REDUCED_DIM}-D)")
    print(f"  • cifar100_ae{REDUCED_DIM}.npz        - Autoencoder features ({REDUCED_DIM}-D)")
    print("\nSaved reducer models:")
    print("  • models/reducers/pca_model.joblib")
    print("  • models/reducers/umap_model.joblib")
    print("  • models/reducers/autoencoder_reducer.pth")
    print("="*60 + "\n")


if __name__ == "__main__":
    save_data()