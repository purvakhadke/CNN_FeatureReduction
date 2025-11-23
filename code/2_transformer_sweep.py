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
TRANSFORMER_EPOCHS = 20    # Transformer training epochs
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

# Dimensions to test (the d_model sweep for Transformer)
TRANSFORMER_DIMS = [1024, 512, 256, 128, 64, 32]
NUM_HEADS = 8  # Number of attention heads
NUM_LAYERS = 2  # Number of transformer encoder layers
FFN_DIM_MULTIPLIER = 4  # Feedforward dimension = d_model * multiplier

# --- A. Transformer Model ---
class TransformerEncoder(nn.Module):
    def __init__(self, d_model):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # Input projection: 2048 -> d_model
        self.input_proj = nn.Linear(INPUT_DIM, d_model)
        
        # Transformer encoder layers
        # Adjust num_heads if d_model is not divisible by NUM_HEADS
        num_heads = min(NUM_HEADS, d_model) if d_model >= NUM_HEADS else d_model
        # Ensure d_model is divisible by num_heads
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * FFN_DIM_MULTIPLIER,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # Output projection: d_model -> 2048 (for reconstruction)
        self.output_proj = nn.Linear(d_model, INPUT_DIM)
        
    def forward(self, x):
        # x shape: (batch_size, INPUT_DIM)
        # Add sequence dimension (treat as sequence of length 1)
        x = x.unsqueeze(1)  # (batch_size, 1, INPUT_DIM)
        
        # Project to d_model
        encoded = self.input_proj(x)  # (batch_size, 1, d_model)
        
        # Apply transformer encoder
        transformed = self.transformer_encoder(encoded)  # (batch_size, 1, d_model)
        
        # Project back to INPUT_DIM for reconstruction
        reconstructed = self.output_proj(transformed)  # (batch_size, 1, INPUT_DIM)
        
        # Remove sequence dimension
        reconstructed = reconstructed.squeeze(1)  # (batch_size, INPUT_DIM)
        transformed_features = transformed.squeeze(1)  # (batch_size, d_model)
        
        return reconstructed, transformed_features

# --- B. Downstream Classifier Model ---
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, OUTPUT_CLASSES) 

    def forward(self, x):
        return self.fc(x)

# --- C. Training and Evaluation Functions ---

def train_transformer(model, loader, epochs):
    """Trains the Transformer using MSE loss for reconstruction."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            # Transformer training is UN-SUPERVISED (reconstruction task)
            inputs = data[0].to(DEVICE)
            optimizer.zero_grad()
            reconstruction, _ = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # print(f"  Transformer Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.6f}")
    
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
    
    results_time = []
    # --- Dimension Sweep Loop ---
    for d_model in TRANSFORMER_DIMS:
        print(f"\n--- Running Sweep for d_model = {d_model} ---")
        
        # 1. Train Transformer
        transformer_model = TransformerEncoder(d_model).to(DEVICE)
        transformer_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        start_time = time.time()
        final_mse_loss = train_transformer(transformer_model, transformer_train_loader, TRANSFORMER_EPOCHS)
        train_time = time.time() - start_time
        results_time.append(train_time)


        
        results_mse.append(final_mse_loss)
        
        # 2. Extract Transformed Features
        transformer_model.eval()
        with torch.no_grad():
            # Extract *all* features from train and test sets
            _, train_transformed_features = transformer_model(train_features.to(DEVICE))
            _, test_transformed_features = transformer_model(test_features.to(DEVICE))

        # 3. Setup Classifier DataLoaders
        cls_train_dataset = TensorDataset(train_transformed_features, train_labels)
        cls_test_dataset = TensorDataset(test_transformed_features, test_labels)
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 4. Train Classifier
        cls_model = Classifier(d_model).to(DEVICE)
        test_accuracy = train_classifier(cls_model, cls_train_loader, cls_test_loader, CLS_EPOCHS)
        results_acc.append(test_accuracy)
        
    # 5. Plot Results
    save_results_csv('results/transformer_results.csv', TRANSFORMER_DIMS, results_mse, results_acc, results_time)
    plot_results(TRANSFORMER_DIMS, results_mse, results_acc)


def plot_results(dims, mse_losses, accuracies):
    """Generates the required plots for the dimension sweep analysis."""
    dims = np.array(dims)
    
    # ----------------------------------------------------
    # PLOT 1: Reconstruction Loss vs. Transformer Dimension (MSE)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dims, mse_losses, marker='o', linestyle='-', color='tab:blue')
    plt.title('Reconstruction Loss (MSE) vs. Transformer Dimension')
    plt.xlabel('Transformer Dimension ($d_{model}$)')
    plt.ylabel('Final MSE Loss')
    plt.grid(True)
    
    # ----------------------------------------------------
    # PLOT 2: Classification Accuracy vs. Transformer Dimension
    # ----------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(dims, accuracies, marker='o', linestyle='-', color='tab:orange')
    plt.title('Classification Accuracy vs. Transformer Dimension')
    plt.xlabel('Transformer Dimension ($d_{model}$)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    
    # --- Save the figure ---
    plt.tight_layout()
    output_plot_filename = 'results/transformer_sweep_results.png'
    plt.savefig(output_plot_filename) 
    
    print(f"\n--- Plots saved successfully to '{output_plot_filename}' ---")

if __name__ == "__main__":
    main_sweep()