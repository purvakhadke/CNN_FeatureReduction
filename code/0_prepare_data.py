# Step 0: Prep and the raw CIFAR-100 data w the dimensionality reduction methods and save the data/models to reuse for the next time we run the code
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


# Autoencoder for Dimensionality Reduction (not the same as autoencoder classifier)
class DimReductionAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DimReductionAutoencoder, self).__init__()
        
        # Encoder input_dim -> latent_dim
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
        
        # Decoder latent_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  
            # Output gives form [0, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder_reducer(train_data, latent_dim, device, epochs=30, batch_size=128):
    model = DimReductionAutoencoder(32*32*3, latent_dim).to(device)
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
    
    return model


def transform_with_autoencoder(model, data, device, batch_size=256):
    
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
    # save the data and save the models so after the first time we run code it will take less computation time since it doesnt have to load
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models/reducers', exist_ok=True)
    
    # this uses GPU if available
    # I have a macbook so only CPU works, 
    # I did not try Colab so maybe it works maybe it doesn't
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    # Load the CIFAR-100 data
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True)
    
    print("Prep/prepocessing the CIFAR-100 Data")    
    # Flatten the images to 3072D each (32x32x3)
    train_images = trainset.data.reshape(len(trainset.data), -1).astype(np.float32) / 255.0
    train_labels = np.array(trainset.targets)
    test_images = testset.data.reshape(len(testset.data), -1).astype(np.float32) / 255.0
    test_labels = np.array(testset.targets)
    
    print(f"Train shape: {train_images.shape}")
    print(f"Test shape: {test_images.shape}")
    
    # Save the raw data as npz file so don't have to load again next time we run code
    np.savez_compressed(
        '../data/cifar100_raw.npz',
        train_features=train_images,
        train_labels=train_labels,
        test_features=test_images,
        test_labels=test_labels
    )
    

    # PCA Reduction (same way we save that file)
    # PCA is just math (no training), so we can just import PCA fn
    pca = PCA(n_components = REDUCED_DIM, random_state = RANDOM_SEED) # the random seed part was recc by llm and documentation, i put in the config file
    pca.fit(train_images)
    
    train_pca = pca.transform(train_images)
    test_pca = pca.transform(test_images)
    
    pca_variance = pca.explained_variance_ratio_.sum()
    # the variance at this REDUCED_DIM(512) is high which is what we want
    print(f"Variance: {pca_variance:.4f} ({pca_variance*100:.2f}%)")
    
    # Save PCA model and data in npz file
    joblib.dump(pca, '../models/reducers/pca_model.joblib')
    np.savez_compressed(
        f'../data/cifar100_pca{REDUCED_DIM}.npz',
        train_features=train_pca,
        train_labels=train_labels,
        test_features=test_pca,
        test_labels=test_labels,
        variance_explained=pca_variance
    )
    
    # UMAP Reduction
    umap_reducer = UMAP(
        n_components = REDUCED_DIM,
        n_neighbors = 15,
        min_dist = 0.01,
        metric='euclidean',
        random_state=RANDOM_SEED,
        verbose=True
    )
    
    # Fit on the training data
    train_umap = umap_reducer.fit_transform(train_images)
    # Transform test data using the fitted model
    test_umap = umap_reducer.transform(test_images)
    
    # Save the UMAP model and data
    joblib.dump(umap_reducer, '../models/reducers/umap_model.joblib')
    np.savez_compressed(
        f'../data/cifar100_umap{REDUCED_DIM}.npz',
        train_features=train_umap,
        train_labels=train_labels,
        test_features=test_umap,
        test_labels=test_labels
    )
    
    # Autoencoder Reduction
    ae_model = train_autoencoder_reducer(
        train_images, 
        latent_dim=REDUCED_DIM, 
        device=device, 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Transform the data
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
    

if __name__ == "__main__":
    # WHATEVER FILES THESE GENERATE, DO NOT PUSH, data file and models file might be too big
    save_data()