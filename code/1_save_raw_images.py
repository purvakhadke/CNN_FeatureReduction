"""
Save raw CIFAR-10 images as flattened 3072-D vectors
"""
import torchvision
import numpy as np
import os
from config import *

def save_raw_images():
    print("\n" + "="*60)
    print("Saving Raw CIFAR-10 Images (3072-D)")
    print("="*60 + "\n")
    
    if os.path.exists('cifar100-raw.npz'):
        print("Raw image file already exists, loading to verify...")
        data = np.load('cifar100-raw.npz')
        print(f"\nFile contents:")
        print(f"  Train features: {data['train_features'].shape}")
        print(f"  Train labels: {data['train_labels'].shape}")
        print(f"  Test features: {data['test_features'].shape}")
        print(f"  Test labels: {data['test_labels'].shape}")
        return
    
    # Load CIFAR-10 (raw numpy arrays)
    print("Downloading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True
    )
    
    # Flatten images: (N, 32, 32, 3) -> (N, 3072)
    print("\nFlattening images...")
    train_images = trainset.data.reshape(len(trainset.data), -1).astype(np.float32) / 255.0
    train_labels = np.array(trainset.targets)
    
    test_images = testset.data.reshape(len(testset.data), -1).astype(np.float32) / 255.0
    test_labels = np.array(testset.targets)
    
    if TRAIN_SAMPLE_SIZE is not None:
        train_images = train_images[:TRAIN_SAMPLE_SIZE]
        train_labels = train_labels[:TRAIN_SAMPLE_SIZE]
    if TEST_SAMPLE_SIZE is not None:
        test_images = test_images[:TEST_SAMPLE_SIZE]
        test_labels = test_labels[:TEST_SAMPLE_SIZE]


    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    
    # Save
    print("\nSaving to cifar100-raw.npz...")
    np.savez_compressed(
        'cifar10-raw',
        train_features=train_images,
        train_labels=train_labels,
        test_features=test_images,
        test_labels=test_labels
    )
    
    print("✅ Raw images saved successfully!\n")

def main():
    save_raw_images()

if __name__ == "__main__":
    main()