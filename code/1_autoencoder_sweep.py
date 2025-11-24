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


# Autoencoder model to encode and decode with layer
# should we remove the layers to match transformers? or add layers in transformers to match autoencoders? (or doesnt matter?)
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        ####
        #### QUESTION I HAVE
        #### INSTEAD OF A, can we do B?
        #### A: 2048 -> 1024 -> 512 -> latent_dim
        #### B: 2048 -> latent_dim (compress directly from 2048 to requested latent dim)
        #### Bc we do B for transformers, idk if we should keep encoder and transformers the same 
        

        # Encoder: 2048 -> 1024 -> 512 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim) 
        )
        
        
        # decoder is only for training, testing will directly use classifier after encoder 
        # Decoder: latent_dim  -> 512 -> 1024 -> 2048 
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

# Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, len(CLASSES)) 

    def forward(self, x):
        return self.fc(x)


# Train autoencoder using MSE loss
def train_autoencoder(model, loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in loader:
            # Autoencoder training is UN-SUPERVISED
            inputs = data[0]
            optimizer.zero_grad()
            reconstruction, _ = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(loader) # Return final average loss

#Train classigider using CorssEntoru loss
def train_classifier(model, train_loader, test_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features, labels
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Run classiger on test data and get total accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features, labels
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = (correct / total) * 100
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
    interpretability_metrics = []

    # --- Dimension Sweep Loop ---
    for d_latent in DIMENSIONS_TO_COMPRESS_TO:
        print(f"\n--- Running Sweep for D_latent = {d_latent} ---")
        
        # 1. Train Autoencoder
        ae_model = Autoencoder(d_latent)
        ae_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        # we need to see how long it took to train bc we are using time as the metric for efficiency
        start_time = time.time()
        final_mse_loss = train_autoencoder(ae_model, ae_train_loader, EPOCHS)
        train_time = time.time() - start_time
        
        
        results_mse.append(final_mse_loss)
        results_time.append(train_time)

        # 2. Extract Latent Features
        ae_model.eval()
        with torch.no_grad():
            # Extract *all* features from train and test sets
            train_latent_features = ae_model.encoder(train_features)
            test_latent_features = ae_model.encoder(test_features)

        # 3. Setup Classifier DataLoaders
        cls_train_dataset = TensorDataset(train_latent_features, train_labels)
        cls_test_dataset = TensorDataset(test_latent_features, test_labels)
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 4. Train Classifier
        cls_model = Classifier(d_latent)
        test_accuracy = train_classifier(cls_model, cls_train_loader, cls_test_loader, EPOCHS_classifier)
        results_acc.append(test_accuracy)
        
        # 5. RUN INTERPRETABILITY FOR THIS DIMENSION
        metrics = run_interpretability_analysis(
            ae_model, train_features, test_features, test_labels, 
            f"Autoencoder-{d_latent}D", d_latent
        )
        interpretability_metrics.append(metrics)

    plot_interpretability_trends(DIMENSIONS_TO_COMPRESS_TO, interpretability_metrics, 'autoencoder')

    # 6. Plot Results
    save_results_csv('../results/autoencoder_results.csv', DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, results_time)
    plot_results(
        dims=DIMENSIONS_TO_COMPRESS_TO,
        mse_losses=results_mse,
        accuracies=results_acc,
        model_name="autoencoder",
        x_label="Latent Dimension ($D_{latent}$)"
    )

def main():
    main_sweep()

if __name__ == "__main__":
    main()
