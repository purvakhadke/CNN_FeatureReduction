import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os
import time

from config import *

# --- CONFIG CHECKS ---
if SAMPLE_SIZE is not None:
    TRAIN_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.9)
    TEST_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.1)
else:
    TRAIN_SAMPLE_SIZE = None
    TEST_SAMPLE_SIZE = None

# --- A. Patch Transformer Autoencoder ---
class PatchTransformerAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(PatchTransformerAutoencoder, self).__init__()
        
        # 1. ENCODER PARTS
        # Project patch (64 dim) up to internal transformer dimension (128 dim)
        self.patch_embedding = nn.Linear(PATCH_SIZE, TRANSFORMER_INTERNAL_DIM)
        
        # Learnable positional encoding (so it knows Patch 1 is different from Patch 32)
        self.pos_embedding = nn.Parameter(torch.randn(1, PATCH_COUNT, TRANSFORMER_INTERNAL_DIM))
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_INTERNAL_DIM,
            nhead=4,
            dim_feedforward=TRANSFORMER_INTERNAL_DIM * 4,
            dropout=0.1,
            activation='gelu', # GELU is standard for Transformers
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 2. BOTTLENECK (COMPRESSION)
        # Flatten the sequence: (32 * 128) -> latent_dim
        self.compressor = nn.Linear(PATCH_COUNT * TRANSFORMER_INTERNAL_DIM, latent_dim)
        
        # 3. DECODER (RECONSTRUCTION)
        # latent_dim -> 2048 (Original Input Size)
        self.decompressor = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, INPUT_DIM)
        )
        
    def forward(self, x):
        # x shape: (Batch, 2048)
        batch_size = x.size(0)
        
        # --- Step 1: Create Sequence (Patching) ---
        # Reshape to (Batch, 32, 64)
        x_seq = x.view(batch_size, PATCH_COUNT, PATCH_SIZE)
        
        # --- Step 2: Embed & Add Position ---
        # Map 64 -> 128
        x_emb = self.patch_embedding(x_seq) 
        # Add position info
        x_emb = x_emb + self.pos_embedding
        
        # --- Step 3: Transformer Magic ---
        # Shape: (Batch, 32, 128)
        trans_out = self.transformer_encoder(x_emb)
        
        # --- Step 4: Compression ---
        # Flatten: (Batch, 32*128)
        flat = trans_out.reshape(batch_size, -1)
        # Compress to latent_dim
        latent_vector = self.compressor(flat)
        
        # --- Step 5: Reconstruction ---
        reconstructed = self.decompressor(latent_vector)
        
        return reconstructed, latent_vector


# --- B. Classifier Model ---
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Deeper classifier for compressed features
        self.net = nn.Sequential(
             nn.Linear(input_dim, input_dim * 2),
             nn.ReLU(),
             nn.Linear(input_dim * 2, len(CLASSES))
        )

    def forward(self, x):
        return self.net(x)

# --- C. Training Functions ---
def train_transformer(model, loader, epochs):
    criterion = nn.MSELoss()
    # AdamW usually works better for Transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
    
    model.train()
    total_loss = 0.0
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
        total_loss = epoch_loss
            
    return total_loss / len(loader)

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
    print(f"  Classifier Test Accuracy: {accuracy:.2f}%")
    return accuracy

# --- D. Main Sweep ---
def main_sweep():
    try:
        data = np.load(FEATURE_FILE)
        if TRAIN_SAMPLE_SIZE is not None:
            train_features = torch.from_numpy(data['train_features'][:TRAIN_SAMPLE_SIZE]).float()
            train_labels = torch.from_numpy(data['train_labels'][:TRAIN_SAMPLE_SIZE]).long()
            test_features = torch.from_numpy(data['test_features'][:TEST_SAMPLE_SIZE]).float()
            test_labels = torch.from_numpy(data['test_labels'][:TEST_SAMPLE_SIZE]).long()
        else:
            train_features = torch.from_numpy(data['train_features']).float()
            train_labels = torch.from_numpy(data['train_labels']).long()
            test_features = torch.from_numpy(data['test_features']).float()
            test_labels = torch.from_numpy(data['test_labels']).long()
    except FileNotFoundError:
        print("Feature file not found.")
        sys.exit(1)

    train_data = TensorDataset(train_features, train_labels)
    print(f"Loaded dataset: Train {len(train_features)}")

    results_mse = []
    results_acc = []
    results_time = []
    interpretability_metrics = []

    for latent_dim in DIMENSIONS_TO_COMPRESS_TO:
        print(f"\n--- Running Transformer Sweep for Latent Dim = {latent_dim} ---")
        
        model_path = f'../models/transformer_patch_{latent_dim}d.pth'
        os.makedirs('../models/', exist_ok=True)
        
        model = PatchTransformerAutoencoder(latent_dim)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state'])
            final_mse = checkpoint['final_mse_loss']
            t_time = checkpoint['train_time']
            print(f"  Loaded pre-trained model.")
        else:
            start_t = time.time()
            final_mse = train_transformer(model, train_loader, EPOCHS)
            t_time = time.time() - start_t
            torch.save({
                'model_state': model.state_dict(), 
                'final_mse_loss': final_mse,
                'train_time': t_time
            }, model_path)
            print(f"  Training done.")
            
        results_mse.append(final_mse)
        results_time.append(t_time)
        
        model.eval()
        with torch.no_grad():
            _, train_latents = model(train_features)
            _, test_latents = model(test_features)
            
        cls_train_ds = TensorDataset(train_latents, train_labels)
        cls_test_ds = TensorDataset(test_latents, test_labels)
        cls_train_loader = DataLoader(cls_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        cls_model = Classifier(latent_dim)
        acc = train_classifier(cls_model, cls_train_loader, cls_test_loader, EPOCHS_classifier)
        results_acc.append(acc)

        metrics = run_interpretability_analysis(
            model, train_features, test_features, test_labels,
            "Transformer", latent_dim
        )
        interpretability_metrics.append(metrics)

    plot_interpretability_trends(DIMENSIONS_TO_COMPRESS_TO, interpretability_metrics, 'transformer')
    save_results_csv('../patched_code_results/transformer_results.csv', DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, results_time)
    plot_results(DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, "transformer", "Latent Dimension")

if __name__ == "__main__":
    main_sweep()