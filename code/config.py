import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

# ============= Fashion-MNIST Configuration =============
SAMPLE_SIZE = None  # Use full dataset (60K train, 10K test)
EPOCHS = 20
EPOCHS_classifier = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# Fashion-MNIST specific settings
FEATURE_FILE = 'fashion_mnist-resnet50.npz'
INPUT_DIM = 2048  # ResNet50 output

# 10 fashion classes
CLASSES = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
)

# Dimensions to test
DIMENSIONS_TO_COMPRESS_TO = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# ============= Same utility functions as other configs =============
# (Copy from config_cifar100.py or config_stanford_cars.py)

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
    output_plot_filename = f'../results/{method_name}_sweep_results.png'
    plt.savefig(output_plot_filename)
    print(f"\n--- Plots saved to '{output_plot_filename}' ---")

def calculate_class_separation(features, labels):
    """Calculate average distance between class centroids."""
    centroids = []
    for i in range(10):  # 10 classes
        class_features = features[labels == i]
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
    """Create t-SNE visualization."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    print(f"  Computing t-SNE for {method_name}-{dimension}D...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_np)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = labels_np == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=CLASSES[i], 
                   alpha=0.6, s=20, edgecolors='none')
    
    plt.title(f'{method_name}: t-SNE Projection ({dimension}D)\nSilhouette: {silhouette:.4f}, Separation: {separation:.2f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best', framealpha=0.9, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_filename = f'../results/{method_name.lower()}_tsne_{dimension}d.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to '{output_filename}'")

def run_interpretability_analysis(model, train_features, test_features, test_labels, method_name, dimension, create_plot=True):
    """Run interpretability analysis."""
    from sklearn.metrics import silhouette_score
    
    # Extract features
    model.eval()
    with torch.no_grad():
        if 'Autoencoder' in method_name:
            reduced_features = model.encoder(test_features)
        else:  # Transformer
            _, reduced_features = model(test_features)
    
    reduced_np = reduced_features.cpu().numpy()
    labels_np = test_labels.numpy()
    
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