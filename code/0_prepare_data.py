"""
Step 0: Save raw CIFAR-100 data as flattened vectors and apply PCA
"""
import torchvision
import numpy as np
import os
from sklearn.decomposition import PCA
from config import *

def save_data():
    print("\n" + "="*60)
    print("Step 0: Preparing CIFAR-100 Data")
    print("="*60 + "\n")
    
    os.makedirs('../data', exist_ok=True)
    
    # Load CIFAR-100
    print("Loading CIFAR-100...")
    trainset = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True
    )
    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True
    )
    
    # Flatten images: (N, 32, 32, 3) -> (N, 3072)
    print("Flattening images to 3072-D vectors...")
    train_images = trainset.data.reshape(len(trainset.data), -1).astype(np.float32) / 255.0
    train_labels = np.array(trainset.targets)
    
    test_images = testset.data.reshape(len(testset.data), -1).astype(np.float32) / 255.0
    test_labels = np.array(testset.targets)
    
    # Apply sample size if testing
    if TRAIN_SAMPLE_SIZE is not None:
        print(f"Using sample size: {SAMPLE_SIZE}")
        train_images = train_images[:TRAIN_SAMPLE_SIZE]
        train_labels = train_labels[:TRAIN_SAMPLE_SIZE]
        test_images = test_images[:TEST_SAMPLE_SIZE]
        test_labels = test_labels[:TEST_SAMPLE_SIZE]
    
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Save raw data
    print(f"\nSaving raw 3072-D data...")
    np.savez_compressed(
        '../data/cifar100_raw.npz',
        train_features=train_images,
        train_labels=train_labels,
        test_features=test_images,
        test_labels=test_labels
    )
    print("✅ Raw data saved to '../data/cifar100_raw.npz'")
    
    # Apply PCA
    print(f"\nApplying PCA: 3072-D → {PCA_COMPONENTS}-D...")
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(train_images)
    
    train_pca = pca.transform(train_images)
    test_pca = pca.transform(test_images)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
    
    # Save PCA data
    print(f"Saving PCA {PCA_COMPONENTS}-D data...")
    np.savez_compressed(
        f'../data/cifar100_pca{PCA_COMPONENTS}.npz',
        train_features=train_pca,
        train_labels=train_labels,
        test_features=test_pca,
        test_labels=test_labels,
        variance_explained=variance_explained
    )
    print(f"✅ PCA data saved to '../data/cifar100_pca{PCA_COMPONENTS}.npz'\n")

if __name__ == "__main__":
    save_data()