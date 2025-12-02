"""
Configuration for CIFAR-100 Classification with/without PCA
Comparing: ResNet, Transformer, Autoencoder as classifiers
"""

# ============================================================
# SAMPLE SIZE CONFIGURATION
# ============================================================
SAMPLE_SIZE = None  # Use full dataset for final results
# SAMPLE_SIZE = 5000  # Uncomment for quick testing

EPOCHS = 30  # Training epochs for Transformer/Autoencoder classifiers
RESNET_EPOCH = 10  # Epochs for ResNet (less needed with pretrained weights)

# ============================================================
# Derived sample sizes
# ============================================================
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
# PCA CONFIGURATION
# ============================================================
if TRAIN_SAMPLE_SIZE is not None:
    PCA_COMPONENTS = min(512, TRAIN_SAMPLE_SIZE - 1)
else:
    PCA_COMPONENTS = 512  # More components for CIFAR-100

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# ResNet specific
RESNET_IMAGE_SIZE = 96  # Resize for faster training (instead of 224)

# Transformer specific
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1

# Autoencoder classifier specific
AUTOENCODER_HIDDEN_DIMS = [512, 256, 128]