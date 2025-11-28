import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import time

# ============= Stanford Cars Configuration =============
SAMPLE_SIZE = None  # Use full dataset
EPOCHS = 20
EPOCHS_classifier = 15  # Slightly more epochs for 196 classes
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# Stanford Cars specific settings
FEATURE_FILE = 'stanford_cars-resnet50.npz'
INPUT_DIM = 2048  # ResNet50 output

# 196 car model classes
CLASSES = None  # Loaded at runtime

# Dimensions to test
DIMENSIONS_TO_COMPRESS_TO = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# ============= Helper function to load classes =============
def load_stanford_cars_classes():
    """Load Stanford Cars class names from saved features"""
    if os.path.exists(FEATURE_FILE):
        data = np.load(FEATURE_FILE, allow_pickle=True)
        class_names = data['class_names']
        data.close()
        return tuple(class_names)
    else:
        # Default placeholder (196 classes)
        return tuple([f"car_model_{i}" for i in range(196)])

# Load classes
CLASSES = load_stanford_cars_classes()

# ============= Same utility functions =============

def plot_interpretability_trends(dims, metrics_list, method_name):
    """Plot how interpretability changes across dimensions."""
    silhouettes = [m['silhouette'] for m in metrics_list]
    separations = [m['separation'] for m in metrics_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(dims, silhouettes, marker='o', linewidth=2, color='tab:purple')
    axes[0].set_title('Silhouette Score vs Dimension', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(dims, separations, marker='o', linewidth=2, color='tab:green')
    axes[1].set_title('Class Separation vs Dimension', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Average Inter-Class Distance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../results/{method_name}_interpretability_trends.png', dpi=300)
    print(f"  ✅ Interpretability trends saved")

def plot_results(dims, mse_losses, accuracies, 
                 model_name="autoencoder",
                 x_label="Latent Dimension ($D_{latent}$)"):
    """Generate sweep analysis plots."""
    dims = np.array(dims)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dims, mse_losses, marker='o', linestyle='-', color='tab:blue')
    plt.title(f'Reconstruction Loss (MSE) vs. {x_label}')
    plt.xlabel(x_label)
    plt.ylabel('Final MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(dims, accuracies, marker='o', linestyle='-', color='tab:orange')
    plt.title(f'Classification Accuracy vs. {x_label}')
    plt.xlabel(x_label)
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    output_plot_filename = f'../results/{model_name}_sweep_results.png'
    plt.savefig(output_plot_filename)
    print(f"\n--- Plots saved to '{output_plot_filename}' ---")

def calculate_class_separation(features, labels):
    """Calculate average distance between class centroids."""
    unique_labels = np.unique(labels)
    centroids = []
    
    # Sample classes for efficiency (196 classes is a lot!)
    sample_size = min(50, len(unique_labels))
    sampled_labels = np.random.choice(unique_labels, sample_size, replace=False)
    
    for label in sampled_labels:
        class_features = features[labels == label]
        if len(class_features) > 0:
            centroids.append(np.mean(class_features, axis=0))
    
    centroids = np.array(centroids)
    
    distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0

def create_tsne_visualization(features_np, labels_np, method_name, dimension, silhouette, separation):
    """Create t-SNE visualization - sample subset for 196 classes."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # For 196 classes, sample heavily for visualization
    max_samples_per_class = 20  # Only 20 samples per class
    indices = []
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        class_indices = np.where(labels_np == label)[0]
        if len(class_indices) > max_samples_per_class:
            class_indices = np.random.choice(class_indices, max_samples_per_class, replace=False)
        indices.extend(class_indices)
    
    features_sampled = features_np[indices]
    labels_sampled = labels_np[indices]
    
    print(f"  Computing t-SNE for {method_name}-{dimension}D (sampled {len(features_sampled)} points)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sampled)
    
    # Plot with many classes - use very small markers, no legend
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_sampled, cmap='tab20c', 
                         alpha=0.5, s=5, edgecolors='none')
    
    plt.title(f'{method_name}: t-SNE Projection ({dimension}D) - 196 Car Models\nSilhouette: {silhouette:.4f}, Separation: {separation:.2f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, label='Car Model ID')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_filename = f'../results/{method_name.lower()}_tsne_{dimension}d.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to '{output_filename}'")

def run_interpretability_analysis(model, train_features, test_features, test_labels, method_name, dimension, create_plot=True):
    """Run interpretability analysis."""
    from sklearn.metrics import silhouette_score
    
    # Sample heavily for silhouette score (196 classes, ~8K test samples)
    max_samples = 3000
    if len(test_features) > max_samples:
        indices = np.random.choice(len(test_features), max_samples, replace=False)
        test_features_sample = test_features[indices]
        test_labels_sample = test_labels[indices]
    else:
        test_features_sample = test_features
        test_labels_sample = test_labels
    
    # Extract features
    model.eval()
    with torch.no_grad():
        if 'Autoencoder' in method_name:
            reduced_features = model.encoder(test_features_sample)
        else:  # Transformer
            _, reduced_features = model(test_features_sample)
    
    reduced_np = reduced_features.cpu().numpy()
    labels_np = test_labels_sample.numpy()
    
    # Calculate metrics
    silhouette = silhouette_score(reduced_np, labels_np)
    separation = calculate_class_separation(reduced_np, labels_np)
    
    print(f"  [{dimension}D] Silhouette: {silhouette:.4f}, Separation: {separation:.2f}")
    
    # Create visualization
    if create_plot:
        create_tsne_visualization(reduced_np, labels_np, method_name, dimension, silhouette, separation)
    
    return {'dimension': dimension, 'silhouette': silhouette, 'separation': separation}

def save_results_csv(filename, dims, mse_losses, accuracies, times):
    """Save results to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Latent_Dim', 'MSE_Loss', 'Accuracy', 'Training_Time_sec'])
        for d, mse, acc, t in zip(dims, mse_losses, accuracies, times):
            writer.writerow([d, mse, acc, t])
    
    print(f"\n--- Results saved to {filename} ---")