"""
Centralized Configuration for Dimensionality Reduction Project
"""

# ============================================================
# SAMPLE SIZE CONFIGURATION (FOR TESTING)
# ============================================================
# Set to None to use full dataset, or set to a number (e.g., 2000) for quick testing
# 
SAMPLE_SIZE = None  # Use full dataset (50,000 train, 10,000 test)
EPOCHS = 30  # Full training
RESNET_EPOCH = 10  # More epochs for baseline
DIMENSIONS_TO_COMPRESS_TO = [10, 25, 50, 100]  # Add 25 back in

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
NUM_CLASSES = 100

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




