"""
Configuration for CIFAR-100 Classification Experiment
Comparing: 
  - Dimensionality Reduction: Raw, PCA, UMAP, Autoencoder
  - Classifiers: ResNet, Transformer, Autoencoder
"""

# ============================================================
# SAMPLE SIZE CONFIGURATION
# ============================================================
SAMPLE_SIZE = None  # Use full dataset for final results
# SAMPLE_SIZE = 5000  # Uncomment for quick testing

# Derived sample sizes
if SAMPLE_SIZE is not None:
    TRAIN_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.9)
    TEST_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.1)
else:
    TRAIN_SAMPLE_SIZE = None
    TEST_SAMPLE_SIZE = None

# ============================================================
# DATA CONFIGURATION
# ============================================================
NUM_CLASSES = 100  # CIFAR-100
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
FLATTENED_DIM = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS  # 3072

# ============================================================
# DIMENSIONALITY REDUCTION CONFIGURATION
# ============================================================
# Target dimension for all reduction methods (PCA, UMAP, Autoencoder)
if TRAIN_SAMPLE_SIZE is not None:
    REDUCED_DIM = min(512, TRAIN_SAMPLE_SIZE - 1)
else:
    REDUCED_DIM = 512

# For backward compatibility
PCA_COMPONENTS = REDUCED_DIM

# UMAP specific parameters
UMAP_N_NEIGHBORS = 15      # Number of neighbors (default: 15)
UMAP_MIN_DIST = 0.1        # Minimum distance (default: 0.1)

# Autoencoder reducer specific
AE_REDUCER_EPOCHS = 30     # Epochs for training dim reduction autoencoder

# ============================================================
# CLASSIFIER TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Epochs for classifiers
EPOCHS = 30                # Transformer and Autoencoder classifier epochs
RESNET_EPOCH = 10          # ResNet epochs (less needed with pretrained)

# ResNet specific
RESNET_IMAGE_SIZE = 96     # Resize for faster training (instead of 224)

# Transformer classifier specific
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1

# Autoencoder classifier specific
AUTOENCODER_HIDDEN_DIMS = [512, 256, 128]

# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_SAMPLE_SIZE = 10000 #2000    # Samples for t-SNE visualization

# ============================================================
# INPUT TYPES FOR EXPERIMENTS
# ============================================================
INPUT_TYPES = ['Raw', f'PCA-{REDUCED_DIM}', f'UMAP-{REDUCED_DIM}', f'AE-{REDUCED_DIM}']
INPUT_FILES = {
    'Raw': '../data/cifar100_raw.npz',
    f'PCA-{REDUCED_DIM}': f'../data/cifar100_pca{REDUCED_DIM}.npz',
    f'UMAP-{REDUCED_DIM}': f'../data/cifar100_umap{REDUCED_DIM}.npz',
    f'AE-{REDUCED_DIM}': f'../data/cifar100_ae{REDUCED_DIM}.npz'
}