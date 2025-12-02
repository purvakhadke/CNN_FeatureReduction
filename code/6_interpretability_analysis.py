"""
RQ3: Analyze interpretability using t-SNE visualizations and clustering metrics
"""
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def calculate_class_separation(features, labels):
    """Calculate average distance between class centroids"""
    centroids = []
    for i in range(10):
        class_features = features[labels == i]
        centroids.append(np.mean(class_features, axis=0))
    centroids = np.array(centroids)
    
    distances = []
    for i in range(10):
        for j in range(i+1, 10):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            distances.append(dist)
    
    return np.mean(distances)

def create_tsne_visualization(features, labels, method_name, dimension):
    """Create t-SNE 2D visualization"""
    print(f"  Creating t-SNE for {method_name}-{dimension}D...")
    
    # Subsample to 5000 points for speed (t-SNE is slow)
    n_samples = min(5000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    features_sub = features[indices]
    labels_sub = labels[indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sub)
    
    # Calculate metrics on full dataset
    silhouette = silhouette_score(features, labels)
    separation = calculate_class_separation(features, labels)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        mask = labels_sub == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=CLASSES[i],
                   alpha=0.6, s=20, edgecolors='none')
    
    plt.title(f'{method_name} ({dimension}D)\n'
              f'Silhouette: {silhouette:.4f}, Separation: {separation:.2f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best', framealpha=0.9, fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs('../results/tsne/', exist_ok=True)
    filename = f'../results/tsne/{method_name.lower().replace(" ", "_")}_{dimension}d.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Silhouette: {silhouette:.4f}, Separation: {separation:.2f}")
    return silhouette, separation

def main():
    print("\n" + "="*60)
    print("RQ3: Interpretability Analysis")
    print("="*60 + "\n")
    
    results = []
    
    # 1. PCA-200
    print("Analyzing PCA-200...")
    data = np.load('../data/cifar100-pca200.npz')
    test_features = data['test_features']
    test_labels = data['test_labels']
    
    sil, sep = create_tsne_visualization(test_features, test_labels, 'PCA', 200)
    results.append(('PCA-200', 200, sil, sep))
    
    # 2. Autoencoder compressed features
    print("\nAnalyzing Autoencoder compressed features...")
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded, encoded
    
    pca_features = torch.from_numpy(test_features).float()
    
    for latent_dim in [10, 50, 100]:
        model_path = f'../models/autoencoder_pca200_to_{latent_dim}d.pth'
        if os.path.exists(model_path):
            model = Autoencoder(200, latent_dim)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            with torch.no_grad():
                compressed = model.encoder(pca_features).numpy()
            
            sil, sep = create_tsne_visualization(compressed, test_labels, 
                                                'Autoencoder', latent_dim)
            results.append((f'PCA→AE-{latent_dim}', latent_dim, sil, sep))
    
    # 3. Transformer compressed features
    print("\nAnalyzing Transformer compressed features...")
    
    PATCH_COUNT = 10
    PATCH_SIZE = 20
    TRANSFORMER_DIM = 64
    
    class TransformerAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(TransformerAutoencoder, self).__init__()
            self.patch_embedding = nn.Linear(PATCH_SIZE, TRANSFORMER_DIM)
            self.pos_embedding = nn.Parameter(torch.randn(1, PATCH_COUNT, TRANSFORMER_DIM))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=TRANSFORMER_DIM, nhead=4,
                dim_feedforward=TRANSFORMER_DIM * 4,
                dropout=0.1, activation='gelu', batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.compressor = nn.Linear(PATCH_COUNT * TRANSFORMER_DIM, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        def forward(self, x):
            batch_size = x.size(0)
            x_seq = x.view(batch_size, PATCH_COUNT, PATCH_SIZE)
            x_emb = self.patch_embedding(x_seq)
            x_emb = x_emb + self.pos_embedding
            trans_out = self.transformer_encoder(x_emb)
            flat = trans_out.reshape(batch_size, -1)
            latent = self.compressor(flat)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
    
    for latent_dim in [10, 50, 100]:
        model_path = f'../models/transformer_pca200_to_{latent_dim}d.pth'
        if os.path.exists(model_path):
            model = TransformerAutoencoder(200, latent_dim)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            with torch.no_grad():
                _, compressed = model(pca_features)
                compressed = compressed.numpy()
            
            sil, sep = create_tsne_visualization(compressed, test_labels,
                                                'Transformer', latent_dim)
            results.append((f'PCA→TF-{latent_dim}', latent_dim, sil, sep))
    
    # Summary table
    print("\n" + "="*70)
    print("INTERPRETABILITY METRICS SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Dimension':<12} {'Silhouette':<12} {'Separation':<12}")
    print("-"*70)
    for method, dim, sil, sep in results:
        print(f"{method:<20} {dim:<12} {sil:<12.4f} {sep:<12.2f}")
    print("="*70)
    
    # Save to CSV
    import csv
    with open('../results/interpretability_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Dimension', 'Silhouette_Score', 'Class_Separation'])
        for method, dim, sil, sep in results:
            writer.writerow([method, dim, sil, sep])
    
    print("\n✅ Interpretability analysis complete!")
    print("   - t-SNE plots saved to '../results/tsne/'")
    print("   - Metrics saved to '../results/interpretability_metrics.csv'\n")

if __name__ == "__main__":
    main()