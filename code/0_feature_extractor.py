import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import urllib.request
import tarfile
import scipy.io
from pathlib import Path

def download_stanford_cars():
    """
    Download Stanford Cars dataset from alternative source.
    Using Kaggle mirror since official Stanford site is down.
    """
    data_dir = './data/stanford_cars'
    os.makedirs(data_dir, exist_ok=True)
    
    print("="*70)
    print("Stanford Cars Dataset - Manual Download Required")
    print("="*70)
    print("The official Stanford site is down.")
    print("You need to download from an alternative source.")
    print()
    print("="*70)
    print("OPTION 1: Kaggle (Easiest - Recommended)")
    print("="*70)
    print()
    print("1. Go to: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
    print("2. Click 'Download' (free Kaggle account needed)")
    print("3. You'll get 'archive.zip' (~1.9GB)")
    print()
    print("4. Extract it:")
    print(f"   unzip archive.zip -d {data_dir}")
    print()
    print("5. Your folder should have:")
    print(f"   {data_dir}/cars_train/    (8,144 images)")
    print(f"   {data_dir}/cars_test/     (8,041 images)")
    print(f"   {data_dir}/devkit/        (metadata)")
    print()
    print("="*70)
    print("OPTION 2: Direct Download (No account needed)")
    print("="*70)
    print()
    print("1. Go to: https://ai.stanford.edu/~jkrause/cars/car_dataset.html")
    print("2. Download:")
    print("   - Training images: cars_train.tgz")
    print("   - Testing images: cars_test.tgz")
    print("   - Devkit: cars_devkit.tgz")
    print()
    print("3. Extract all three to:")
    print(f"   {data_dir}/")
    print()
    print("="*70)
    print("After downloading, run this script again!")
    print("="*70)
    
    return data_dir

def load_stanford_cars_features():
    """
    Extract ResNet50 features from Stanford Cars dataset.
    196 fine-grained car models - MUCH harder than "car" vs "truck"!
    """
    print("="*70)
    print("Extracting Stanford Cars Features")
    print("="*70)
    print("Stanford Cars: 196 fine-grained car models")
    print("Challenge: Distinguish '2012 Tesla Model S' from '2012 BMW M3'")
    print("ResNet50 only knows 'car' - not specific models!")
    print()
    
    data_dir = './data/stanford_cars'
    
    # Check if data exists
    train_dir = os.path.join(data_dir, 'cars_train')
    test_dir = os.path.join(data_dir, 'cars_test')
    devkit_dir = os.path.join(data_dir, 'devkit')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        download_stanford_cars()
        print("\n⚠️  Data not found! Please download first, then run again.")
        return
    
    print("✓ Found Stanford Cars data")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    print()
    
    # Transform for ResNet50
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load metadata to get class names
    try:
        # Try loading from devkit
        mat_path = os.path.join(devkit_dir, 'cars_meta.mat')
        if os.path.exists(mat_path):
            mat = scipy.io.loadmat(mat_path)
            class_names = [item[0] for item in mat['class_names'][0]]
            print(f"✓ Loaded {len(class_names)} class names from metadata")
        else:
            # Fallback: create generic names
            class_names = [f"car_model_{i:03d}" for i in range(196)]
            print("⚠️  Using generic class names")
    except Exception as e:
        print(f"⚠️  Could not load metadata: {e}")
        class_names = [f"car_model_{i:03d}" for i in range(196)]
    
    print(f"\nSample car models:")
    for i in range(0, min(20, len(class_names)), 4):
        samples = [class_names[j] for j in range(i, min(i+4, len(class_names)))]
        print(f"  {', '.join(samples)}")
    print("  ...")
    print()
    
    # Use ImageFolder to load images
    from torchvision.datasets import ImageFolder
    
    print("Loading images with ImageFolder...")
    trainset = ImageFolder(root=train_dir, transform=transform)
    testset = ImageFolder(root=test_dir, transform=transform)
    
    print(f"✓ Train: {len(trainset)} images")
    print(f"✓ Test:  {len(testset)} images")
    print(f"✓ Classes: {len(trainset.classes)}")
    print()
    
    # Load ResNet50
    print("Loading ImageNet-trained ResNet50...")
    print("⚠️  CRITICAL: ResNet50 only learned 'car' as ONE class!")
    print("It has NO idea about:")
    print("  - Different makes (Tesla vs BMW vs Audi)")
    print("  - Different models (Model S vs M3 vs A4)")
    print("  - Different years (2012 vs 2013)")
    print()
    print("This creates a DOMAIN GAP:")
    print("  ImageNet: Coarse categories (car, truck, SUV)")
    print("  Stanford Cars: Fine-grained models (196 specific cars)")
    print()
    print("Expected: Transformers will SIGNIFICANTLY outperform PCA!")
    print()
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    print("Extracting features from car images...")
    
    # Extract training features
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=32,
        shuffle=False, 
        num_workers=2
    )
    
    train_features_list = []
    train_labels_list = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            features = model(images).squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            train_features_list.append(features.cpu().numpy())
            train_labels_list.append(labels.numpy())
            
            if i % 50 == 0:
                print(f"  Train: {i * 32}/{len(trainset)} processed...")
    
    train_features = np.concatenate(train_features_list)
    train_labels = np.concatenate(train_labels_list)
    print(f"✓ Train features: {train_features.shape}")
    
    # Extract test features
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=32,
        shuffle=False, 
        num_workers=2
    )
    
    test_features_list = []
    test_labels_list = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            features = model(images).squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            test_features_list.append(features.cpu().numpy())
            test_labels_list.append(labels.numpy())
            
            if i % 50 == 0:
                print(f"  Test: {i * 32}/{len(testset)} processed...")
    
    test_features = np.concatenate(test_features_list)
    test_labels = np.concatenate(test_labels_list)
    print(f"✓ Test features: {test_features.shape}")
    
    # Save features
    np.savez_compressed(
        'stanford_cars-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        class_names=class_names
    )
    
    print("\n" + "="*70)
    print("✅ Stanford Cars features extracted!")
    print("="*70)
    print(f"Train samples: {len(train_features):,}")
    print(f"Test samples:  {len(test_features):,}")
    print(f"Classes: {len(class_names)} (196 car models)")
    print()
    print("Expected Results (Fine-grained classification is HARD!):")
    print("  PCA:         35-45% (linear methods struggle with subtle differences)")
    print("  Autoencoder: 50-60% (+15%)")
    print("  Transformer: 55-65% (+20%) ✅ CLEAR ADVANTAGE!")
    print()
    print("="*70)

def main():
    if os.path.exists('stanford_cars-resnet50.npz'):
        print("Stanford Cars features already exist")
        data = np.load('stanford_cars-resnet50.npz', allow_pickle=True)
        print(f"\nStanford Cars File contents:")
        print(f"----Train----")
        print(f"train_features: {data['train_features'].shape}")
        print(f"train_labels: {data['train_labels'].shape}")
        print(f"----Test-----")
        print(f"test_features: {data['test_features'].shape}")
        print(f"test_labels: {data['test_labels'].shape}")
        print(f"----Classes----")
        print(f"Number of classes: {len(data['class_names'])}")
        print(f"Sample classes: {list(data['class_names'][:5])}")
        data.close()
    else:
        load_stanford_cars_features()

if __name__ == "__main__":
    main()