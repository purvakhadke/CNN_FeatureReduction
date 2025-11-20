import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
DEVICE = torch.device("cpu") 
INPUT_DIM = 2048
OUTPUT_CLASSES = 10
AE_EPOCHS = 20    # Autoencoder training epochs
CLS_EPOCHS = 10   # Classifier training epochs
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
# --- MUST MATCH THE OUTPUT FILE NAME FROM feature_extractor.py ---
FEATURE_FILE = 'cifar10-resnet50_features_SAMPLE_2000.npz' 

# Dimensions to test (the D_latent sweep)
LATENT_DIMS = [1024, 512, 256, 128, 64, 32] 
# Note: Since we only have 2000 samples, we must reduce the train/test split size.
TRAIN_SIZE = 1500
TEST_SIZE = 500

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
    # --- Data Loading and Splitting ---
    try:
        data = np.load(FEATURE_FILE)
        all_features = torch.from_numpy(data['features']).float()
        all_labels = torch.from_numpy(data['labels']).long()
    except FileNotFoundError:
        print(f"Error: Feature file '{FEATURE_FILE}' not found. Did you run feature_extractor.py first?")
        sys.exit(1)

    full_dataset = TensorDataset(all_features, all_labels)
    # Ensure our split matches the small sample size (e.g., 2000 total)
    train_data, test_data = random_split(full_dataset, [TRAIN_SIZE, len(full_dataset) - TRAIN_SIZE])

    # --- Sweep Storage ---
    results_mse = []
    results_acc = []

    # --- Dimension Sweep Loop ---
    for d_latent in LATENT_DIMS:
        print(f"\n--- Running Sweep for D_latent = {d_latent} ---")
        
        # 1. Train Autoencoder
        ae_model = Autoencoder(d_latent).to(DEVICE)
        ae_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        final_mse_loss = train_autoencoder(ae_model, ae_train_loader, AE_EPOCHS)
        results_mse.append(final_mse_loss)
        
        # 2. Extract Latent Features
        ae_model.eval()
        with torch.no_grad():
            # Extract *all* features from the subsets for classifier training
            train_latent_features = ae_model.encoder(train_data[:][0].to(DEVICE))
            test_latent_features = ae_model.encoder(test_data[:][0].to(DEVICE))

        # 3. Setup Classifier DataLoaders
        cls_train_dataset = TensorDataset(train_latent_features, train_data[:][1])
        cls_test_dataset = TensorDataset(test_latent_features, test_data[:][1])
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 4. Train Classifier
        cls_model = Classifier(d_latent).to(DEVICE)
        test_accuracy = train_classifier(cls_model, cls_train_loader, cls_test_loader, CLS_EPOCHS)
        results_acc.append(test_accuracy)
        
    # 5. Plot Results
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
    
    # --- Resolution Step: Save the figure instead of trying to show it interactively ---
    plt.tight_layout()
    output_plot_filename = 'autoencoder_sweep_results.png'
    plt.savefig(output_plot_filename) 
    
    print(f"\n--- Plots saved successfully to '{output_plot_filename}' ---")

if __name__ == "__main__":
    main_sweep()