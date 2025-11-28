import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os
import time

from config import *

if SAMPLE_SIZE is not None:
    TRAIN_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.9)
    TEST_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.1)
else:
    TRAIN_SAMPLE_SIZE = None
    TEST_SAMPLE_SIZE = None

# --- A. Autoencoder Model (With BatchNorm) ---
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder: 2048 -> 1024 -> 512 -> latent_dim
        # Added BatchNorm1d for better convergence
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim) 
        )
        
        # Decoder: latent_dim -> 512 -> 1024 -> 2048 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, INPUT_DIM) 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# --- B. Classifier Model ---
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, len(CLASSES)) 

    def forward(self, x):
        return self.fc(x)

# --- C. Training Functions ---
def train_autoencoder(model, loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            
    accuracy = (correct / total) * 100
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
        print(f"Error: Feature file '{FEATURE_FILE}' not found.")
        sys.exit(1)

    train_data = TensorDataset(train_features, train_labels)
    print(f"Loaded dataset: Train {len(train_features)} samples")

    results_mse = []
    results_acc = []
    results_time = []  
    interpretability_metrics = []

    for d_latent in DIMENSIONS_TO_COMPRESS_TO:
        print(f"\n--- Running Sweep for D_latent = {d_latent} ---")
        model_path = f'../models/autoencoder_{d_latent}d.pth'
        os.makedirs('../models/', exist_ok=True)
        
        autoencoder_model = Autoencoder(d_latent)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            autoencoder_model.load_state_dict(checkpoint['model_state'])
            final_mse_loss = checkpoint['final_mse_loss']
            train_time = checkpoint['train_time']
            print(f"Loaded pre-trained Autoencoder.")
        else:
            ae_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
            start_time = time.time()
            final_mse_loss = train_autoencoder(autoencoder_model, ae_train_loader, EPOCHS)
            train_time = time.time() - start_time
            torch.save({
                'model_state': autoencoder_model.state_dict(),
                'final_mse_loss': final_mse_loss,
                'train_time': train_time
            }, model_path)
            print(f"Trained Autoencoder.")

        results_mse.append(final_mse_loss)
        results_time.append(train_time)

        autoencoder_model.eval()
        with torch.no_grad():
            train_latent_features = autoencoder_model.encoder(train_features)
            test_latent_features = autoencoder_model.encoder(test_features)

        cls_train_dataset = TensorDataset(train_latent_features, train_labels)
        cls_test_dataset = TensorDataset(test_latent_features, test_labels)
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        cls_model = Classifier(d_latent)
        test_accuracy = train_classifier(cls_model, cls_train_loader, cls_test_loader, EPOCHS_classifier)
        results_acc.append(test_accuracy)
        
        metrics = run_interpretability_analysis(
            autoencoder_model, train_features, test_features, test_labels, 
            f"Autoencoder", d_latent
        )
        interpretability_metrics.append(metrics)

    plot_interpretability_trends(DIMENSIONS_TO_COMPRESS_TO, interpretability_metrics, 'autoencoder')
    save_results_csv('../patched_code_results/autoencoder_results.csv', DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, results_time)
    plot_results(DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, "autoencoder", "Latent Dimension")

if __name__ == "__main__":
    main_sweep()