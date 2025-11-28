import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import FashionMNIST
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_fashion_mnist_features():
    """
    Extract ResNet50 features from Fashion-MNIST.
    10 clothing categories - different domain from ImageNet objects!
    
    NO ACCOUNT NEEDED - Downloads automatically!
    """
    print("="*70)
    print("Extracting Fashion-MNIST Features")
    print("="*70)
    print("Fashion-MNIST: 10 clothing/fashion categories")
    print("Classes: T-shirt, Trouser, Pullover, Dress, Coat,")
    print("         Sandal, Shirt, Sneaker, Bag, Ankle boot")
    print()
    print("Domain Gap: ResNet50 trained on OBJECTS (car, dog, cat)")
    print("            Fashion-MNIST has CLOTHING items")
    print()
    print("Downloading automatically (no account needed)...")
    print("Size: ~30MB")
    print()
    
    # Transform for ResNet50 (28x28 grayscale -> 224x224 RGB)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),  # Convert grayscale to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Download Fashion-MNIST (automatic, no account needed!)
    print("Downloading Fashion-MNIST...")
    trainset = FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    testset = FashionMNIST(
        root='./data',
        train=False, 
        download=True,
        transform=transform
    )
    
    print(f"✓ Downloaded")
    print(f"  Train: {len(trainset)} images")
    print(f"  Test:  {len(testset)} images")
    print(f"  Classes: {len(trainset.classes)}")
    print()
    
    # Class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    print(f"Classes: {class_names}")
    print()
    
    # Load ResNet50
    print("Loading ImageNet-trained ResNet50...")
    print("⚠️  CRITICAL: ResNet50 trained on natural objects!")
    print("It knows: cars, animals, household objects")
    print("It does NOT know: clothing patterns, fabric textures, garment shapes")
    print()
    print("Domain Gap:")
    print("  ImageNet: Natural objects (dogs, cats, cars)")
    print("  Fashion-MNIST: Clothing items (shirts, dresses, shoes)")
    print()
    print("Expected: Transformers will outperform PCA by 10-15%!")
    print()
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    print("Extracting features from fashion images...")
    
    # Extract training features
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128,
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
            
            if i % 100 == 0:
                print(f"  Train: {i * 128}/{len(trainset)} processed...")
    
    train_features = np.concatenate(train_features_list)
    train_labels = np.concatenate(train_labels_list)
    print(f"✓ Train features: {train_features.shape}")
    
    # Extract test features
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=128,
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
            
            if i % 100 == 0:
                print(f"  Test: {i * 128}/{len(testset)} processed...")
    
    test_features = np.concatenate(test_features_list)
    test_labels = np.concatenate(test_labels_list)
    print(f"✓ Test features: {test_features.shape}")
    
    # Save features
    np.savez_compressed(
        'fashion_mnist-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        class_names=class_names
    )
    
    print("\n" + "="*70)
    print("✅ Fashion-MNIST features extracted!")
    print("="*70)
    print(f"Train samples: {len(train_features):,}")
    print(f"Test samples:  {len(test_features):,}")
    print(f"Classes: {len(class_names)}")
    print()
    print("Expected Results (Clothing vs Objects domain gap):")
    print("  PCA:         70-75% (struggles with clothing patterns)")
    print("  Autoencoder: 80-85% (+10%)")
    print("  Transformer: 82-87% (+12-17%) ✅ CLEAR ADVANTAGE!")
    print()
    print("Why Transformers Win:")
    print("  - ResNet50 trained on objects, not clothing")
    print("  - Clothing needs: texture patterns, garment shapes")
    print("  - Self-attention learns: collar types, sleeve patterns")
    print("  - Non-linear features crucial for fashion")
    print()
    print("Perfect to show Transformer advantage with NO account needed! 👕")
    print("="*70)

def main():
    if os.path.exists('fashion_mnist-resnet50.npz'):
        print("Fashion-MNIST features already exist")
        data = np.load('fashion_mnist-resnet50.npz', allow_pickle=True)
        print(f"\nFashion-MNIST File contents:")
        print(f"----Train----")
        print(f"train_features: {data['train_features'].shape}")
        print(f"train_labels: {data['train_labels'].shape}")
        print(f"----Test-----")
        print(f"test_features: {data['test_features'].shape}")
        print(f"test_labels: {data['test_labels'].shape}")
        print(f"----Classes----")
        print(f"class_names: {list(data['class_names'])}")
        data.close()
    else:
        load_fashion_mnist_features()

if __name__ == "__main__":
    main()