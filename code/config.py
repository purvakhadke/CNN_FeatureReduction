"""
Centralized Configuration for Dimensionality Reduction Project
"""

# ============================================================
# SAMPLE SIZE CONFIGURATION (FOR TESTING)
# ============================================================
# Set to None to use full dataset, or set to a number (e.g., 2000) for quick testing
SAMPLE_SIZE = 1000 #None  # Change to 2000 for testing, None for full run

EPOCHS = 3 # 30                    # For autoencoder/transformer training
RESNET_EPOCH = 2 #10
DIMENSIONS_TO_COMPRESS_TO = [10, 50] #[10, 50, 100]


# ============================================================


# Derived sample sizes
if SAMPLE_SIZE is not None:
    TRAIN_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.9)  # 90% for training
    TEST_SAMPLE_SIZE = int(SAMPLE_SIZE * 0.1)   # 10% for testing
else:
    TRAIN_SAMPLE_SIZE = None
    TEST_SAMPLE_SIZE = None

# ============================================================
# DATA CONFIGURATION
# ============================================================
NUM_CLASSES = 10
CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ============================================================
# COMMON TRAINING HYPERPARAMETERS
# ============================================================



if TRAIN_SAMPLE_SIZE is not None:
    # Can't have more components than samples
    PCA_COMPONENTS = min(200, TRAIN_SAMPLE_SIZE - 1)
    print(f"!!!!!PCA_COMPONENTS adjusted to {PCA_COMPONENTS} (limited by sample size)")
else:
    PCA_COMPONENTS = 200  # Full dataset

BATCH_SIZE = 128
LEARNING_RATE = 0.001




