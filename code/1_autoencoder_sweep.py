import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import time

from config import *

# --- Configuration ---
DEVICE = torch.device("cpu") 
INPUT_DIM = 2048
OUTPUT_CLASSES = 10
AE_EPOCHS = 20    # Autoencoder training epochs
CLS_EPOCHS = 10   # Classifier training epochs
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
FEATURE_FILE = 'cifar10-resnet50.npz'  # Just change this line to switch datasets!

TRAIN_SAMPLE_SIZE = None
TEST_SAMPLE_SIZE = None
# set in config.py file
if SAMPLE_SIZE != None:
    TRAIN_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.9)  # 90%
    TEST_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.1)   # 10%
    print("========AYYYYYYYY")
    print("AYYYYYYYY========")
    print("========AYYYYYYYY")
    print("AYYYYYYYY========")
    print("========AYYYYYYYY")
    print(f"SAMPLE SIZE IS {SAMPLE_SIZE}")
    print(f"SAMPLE SIZE IS {SAMPLE_SIZE}")
    print(f"SAMPLE SIZE IS {SAMPLE_SIZE}")
    print("========AYYYYYYYY")
    print("AYYYYYYYY========")
    print("========AYYYYYYYY")
    print("AYYYYYYYY========")
    print("========AYYYYYYYY")

# Dimensions to test (the D_latent sweep)
LATENT_DIMS = [1024, 512, 256, 128, 64, 32] 

# --- A. Autoencoder Model ---
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder: 2048 -> ... -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim) 
        )
        # Decoder: latent_dim -> ... -> 2048
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, INPUT_DIM) 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# --- B. Downstream Classifier Model ---
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, OUTPUT_CLASSES) 

    def forward(self, x):
        return self.fc(x)

# --- C. Training and Evaluation Functions ---

def train_autoencoder(model, loader, epochs):
    """Trains the Autoencoder using MSE loss for reconstruction."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            # Autoencoder training is UN-SUPERVISED
            inputs = data[0].to(DEVICE)
            optimizer.zero_grad()
            reconstruction, _ = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # print(f"  AE Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.6f}")
    
    return running_loss / len(loader) # Return final average loss

def train_classifier(model, train_loader, test_loader, epochs):
    """Trains the Classifier using Cross-Entropy loss for classification."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- Test Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"  Classifier Test Accuracy: {accuracy:.2f}%")
    return accuracy

# --- D. Main Execution and Sweep Loop ---

def main_sweep():
    # --- Data Loading (use pre-split train/test data) ---
    try:
        data = np.load(FEATURE_FILE)
        if TRAIN_SAMPLE_SIZE is not None:
            # Use sampled sizes
            train_features = torch.from_numpy(data['train_features'][:TRAIN_SAMPLE_SIZE]).float()
            train_labels = torch.from_numpy(data['train_labels'][:TRAIN_SAMPLE_SIZE]).long()
            
            test_features = torch.from_numpy(data['test_features'][:TEST_SAMPLE_SIZE]).float()
            test_labels = torch.from_numpy(data['test_labels'][:TEST_SAMPLE_SIZE]).long()
        else:
            # Use full dataset
            train_features = torch.from_numpy(data['train_features']).float()
            train_labels = torch.from_numpy(data['train_labels']).long()
            
            test_features = torch.from_numpy(data['test_features']).float()
            test_labels = torch.from_numpy(data['test_labels']).long()
            
    except FileNotFoundError:
        print(f"Error: Feature file '{FEATURE_FILE}' not found.")
        sys.exit(1)

    train_data = TensorDataset(train_features, train_labels)
    test_data = TensorDataset(test_features, test_labels)
    
    print(f"Loaded dataset from '{FEATURE_FILE}'")
    print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # --- Sweep Storage ---
    results_mse = []
    results_acc = []

    # time it takes for the results
    results_time = []  
    # --- Dimension Sweep Loop ---
    for d_latent in LATENT_DIMS:
        print(f"\n--- Running Sweep for D_latent = {d_latent} ---")
        
        # 1. Train Autoencoder
        ae_model = Autoencoder(d_latent).to(DEVICE)
        ae_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        # we need to see how long it took to train bc we are using time as the metric for efficiency
        start_time = time.time()
        final_mse_loss = train_autoencoder(ae_model, ae_train_loader, AE_EPOCHS)
        train_time = time.time() - start_time
        
        
        results_mse.append(final_mse_loss)
        results_time.append(train_time)

        # 2. Extract Latent Features
        ae_model.eval()
        with torch.no_grad():
            # Extract *all* features from train and test sets
            train_latent_features = ae_model.encoder(train_features.to(DEVICE))
            test_latent_features = ae_model.encoder(test_features.to(DEVICE))

        # 3. Setup Classifier DataLoaders
        cls_train_dataset = TensorDataset(train_latent_features, train_labels)
        cls_test_dataset = TensorDataset(test_latent_features, test_labels)
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 4. Train Classifier
        cls_model = Classifier(d_latent).to(DEVICE)
        test_accuracy = train_classifier(cls_model, cls_train_loader, cls_test_loader, CLS_EPOCHS)
        results_acc.append(test_accuracy)
        
    # 5. Plot Results
    save_results_csv('results/autoencoder_results.csv', LATENT_DIMS, results_mse, results_acc, results_time)
    plot_results(LATENT_DIMS, results_mse, results_acc)


def plot_results(dims, mse_losses, accuracies):
    """Generates the required plots for the dimension sweep analysis."""
    dims = np.array(dims)
    
    # ----------------------------------------------------
    # PLOT 1: Reconstruction Loss vs. Latent Dimension (MSE)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dims, mse_losses, marker='o', linestyle='-', color='tab:blue')
    plt.title('Reconstruction Loss (MSE) vs. Latent Dimension')
    plt.xlabel('Latent Dimension ($D_{latent}$)')
    plt.ylabel('Final MSE Loss')
    plt.grid(True)
    
    # ----------------------------------------------------
    # PLOT 2: Classification Accuracy vs. Latent Dimension
    # ----------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(dims, accuracies, marker='o', linestyle='-', color='tab:orange')
    plt.title('Classification Accuracy vs. Latent Dimension')
    plt.xlabel('Latent Dimension ($D_{latent}$)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    
    # --- Save the figure ---
    plt.tight_layout()
    output_plot_filename = 'results/autoencoder_sweep_results.png'
    plt.savefig(output_plot_filename) 
    
    print(f"\n--- Plots saved successfully to results/'{output_plot_filename}' ---")

if __name__ == "__main__":
    main_sweep()