"""
PCA Baseline for Dimensionality Reduction
Provides traditional baseline comparison for autoencoder and transformer methods.
"""

import numpy as np
import sys
import time
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import *

TRAIN_SAMPLE_SIZE = None
TEST_SAMPLE_SIZE = None

# Set sample sizes from config
if SAMPLE_SIZE != None:
    TRAIN_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.9)
    TEST_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.1)
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

# --- Classifier Model ---
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, len(CLASSES))

    def forward(self, x):
        return self.fc(x)

# --- Training and Evaluation Functions ---

def train_classifier(model, train_loader, test_loader, epochs):
    """Trains the Classifier using Cross-Entropy loss for classification."""
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

    # --- Test Evaluation ---
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
            
    accuracy = 100 * correct / total
    print(f"  Classifier Test Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main Execution and Sweep Loop ---

def main_sweep():
    # --- Data Loading ---
    try:
        data = np.load(FEATURE_FILE)
        
        if TRAIN_SAMPLE_SIZE is not None:
            # Use sampled sizes
            train_features_np = data['train_features'][:TRAIN_SAMPLE_SIZE]
            train_labels_np = data['train_labels'][:TRAIN_SAMPLE_SIZE]
            
            test_features_np = data['test_features'][:TEST_SAMPLE_SIZE]
            test_labels_np = data['test_labels'][:TEST_SAMPLE_SIZE]
        else:
            # Use full dataset
            train_features_np = data['train_features']
            train_labels_np = data['train_labels']
            
            test_features_np = data['test_features']
            test_labels_np = data['test_labels']
            
    except FileNotFoundError:
        print(f"Error: Feature file '{FEATURE_FILE}' not found.")
        sys.exit(1)
    
    print(f"Loaded dataset from '{FEATURE_FILE}'")
    print(f"Train: {len(train_features_np)} samples, Test: {len(test_features_np)} samples")

    # --- Sweep Storage ---
    results_mse = []
    results_acc = []
    results_time = []
    interpretability_metrics = []

    # --- Dimension Sweep Loop ---
    for n_components in DIMENSIONS_TO_COMPRESS_TO:
        print(f"\n--- Running PCA for n_components = {n_components} ---")
        
        # 1. Apply PCA
        start_time = time.time()
        
        pca = PCA(n_components=n_components)
        
        # Fit on training data
        pca.fit(train_features_np)
        
        # Transform both train and test
        train_pca_features = pca.transform(train_features_np)
        test_pca_features = pca.transform(test_features_np)
        
        pca_time = time.time() - start_time
        print(f"  PCA transformation completed in {pca_time:.2f} seconds")
        
        # 2. Calculate reconstruction error (for fair comparison)
        # Reconstruct back to original space
        train_reconstructed = pca.inverse_transform(train_pca_features)
        mse_loss = np.mean((train_features_np - train_reconstructed) ** 2)
        
        results_mse.append(mse_loss)
        results_time.append(pca_time)
        
        print(f"  Reconstruction MSE: {mse_loss:.6f}")
        
        # 3. Convert to PyTorch tensors for classifier
        train_pca_tensor = torch.from_numpy(train_pca_features).float()
        train_labels_tensor = torch.from_numpy(train_labels_np).long()
        test_pca_tensor = torch.from_numpy(test_pca_features).float()
        test_labels_tensor = torch.from_numpy(test_labels_np).long()
        
        # 4. Setup Classifier DataLoaders
        cls_train_dataset = TensorDataset(train_pca_tensor, train_labels_tensor)
        cls_test_dataset = TensorDataset(test_pca_tensor, test_labels_tensor)
        cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        cls_test_loader = DataLoader(cls_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 5. Train Classifier
        cls_model = Classifier(n_components)
        test_accuracy = train_classifier(cls_model, cls_train_loader, cls_test_loader, EPOCHS_classifier)
        results_acc.append(test_accuracy)
        
        # 6. Calculate interpretability metrics
        silhouette = silhouette_score(test_pca_features, test_labels_np)
        separation = calculate_class_separation(test_pca_features, test_labels_np)
        
        print(f"  [{n_components}D] Silhouette: {silhouette:.4f}, Separation: {separation:.2f}")
        
        interpretability_metrics.append({
            'dimension': n_components,
            'silhouette': silhouette,
            'separation': separation
        })
        
        # 7. CREATE T-SNE VISUALIZATION
        # Visualize all tested dimensions
        create_tsne_visualization(
            test_pca_features,
            test_labels_np,
            'PCA',
            n_components,
            silhouette,
            separation
        )

    # 8. Save and Plot Results
    save_results_csv('../patched_code_results/pca_results.csv', DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, results_time)
    plot_results(DIMENSIONS_TO_COMPRESS_TO, results_mse, results_acc, 
                 model_name="pca",
                 x_label="PCA Components")
    plot_interpretability_trends(DIMENSIONS_TO_COMPRESS_TO, interpretability_metrics, 'pca')

    print("\n✅ PCA sweep complete!")


def main():
    main_sweep()


if __name__ == "__main__":
    main()