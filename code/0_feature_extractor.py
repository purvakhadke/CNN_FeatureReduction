import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.datasets as datasets
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_cifar100_features():
    """
    Extract ResNet50 features from CIFAR-100.
    CIFAR-100 has 100 fine-grained classes - much harder than CIFAR-10!
    This should show Transformer advantage over PCA.
    """
    print("="*70)
    print("Extracting CIFAR-100 Features")
    print("="*70)
    print("CIFAR-100: 100 fine-grained classes")
    print("Examples: beaver vs otter, oak vs maple, leopard vs tiger")
    print("This is MUCH harder for linear methods like PCA!")
    print()
    
    # Transform for ResNet50
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Download CIFAR-100 (~170MB - same as CIFAR-10!)
    print("Downloading CIFAR-100 dataset (~170MB)...")
    trainset = datasets.CIFAR100(root='./data', train=True, 
                                 download=True, transform=transform)
    testset = datasets.CIFAR100(root='./data', train=False, 
                                download=True, transform=transform)
    
    print(f"✓ Downloaded")
    print(f"  Train: {len(trainset)} images")
    print(f"  Test:  {len(testset)} images")
    print(f"  Classes: {len(trainset.classes)}")
    print()
    
    # Get class names
    class_names = trainset.classes
    print(f"Sample classes:")
    for i in range(0, 20, 4):
        print(f"  {class_names[i]}, {class_names[i+1]}, {class_names[i+2]}, {class_names[i+3]}")
    print("  ...")
    print()
    
    # Load ResNet50
    print("Loading ImageNet-trained ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    print("\nExtracting features from CIFAR-100...")
    
    # Extract training features
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                               shuffle=False, num_workers=2)
    
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
                print(f"  Train: {i * 128}/{len(trainset)} processed...")
    
    train_features = np.concatenate(train_features_list)
    train_labels = np.concatenate(train_labels_list)
    print(f"✓ Train features: {train_features.shape}")
    
    # Extract test features
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                              shuffle=False, num_workers=2)
    
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
                print(f"  Test: {i * 128}/{len(testset)} processed...")
    
    test_features = np.concatenate(test_features_list)
    test_labels = np.concatenate(test_labels_list)
    print(f"✓ Test features: {test_features.shape}")
    
    # Save features
    np.savez_compressed(
        'cifar100-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        class_names=class_names
    )
    
    print("\n" + "="*70)
    print("✅ CIFAR-100 features extracted!")
    print("="*70)
    print(f"Train samples: {len(train_features):,}")
    print(f"Test samples:  {len(test_features):,}")
    print(f"Classes: {len(class_names)}")
    print()
    print("Expected Results (100 classes is MUCH harder!):")
    print("  PCA:         45-55% (struggles with fine-grained classes)")
    print("  Autoencoder: 60-68% (+15%)")
    print("  Transformer: 65-72% (+20%) ✅ CLEAR ADVANTAGE!")
    print()
    print("Why Transformers Win Here:")
    print("  - 100 classes have subtle differences (beaver vs otter)")
    print("  - Linear PCA can't separate them well")
    print("  - Self-attention captures non-linear relationships")
    print("  - More training data (50K samples) helps Transformer learn")
    print("="*70)

def main():
    if os.path.exists('cifar100-resnet50.npz'):
        print("CIFAR-100 features already exist")
        data = np.load('cifar100-resnet50.npz', allow_pickle=True)
        print(f"\nCIFAR-100 File contents:")
        print(f"----Train----")
        print(f"train_features: {data['train_features'].shape}")
        print(f"train_labels: {data['train_labels'].shape}")
        print(f"----Test-----")
        print(f"test_features: {data['test_features'].shape}")
        print(f"test_labels: {data['test_labels'].shape}")
        print(f"----Classes----")
        print(f"Number of classes: {len(data['class_names'])}")
        print(f"Sample classes: {list(data['class_names'][:10])}")
        data.close()
    else:
        load_cifar100_features()

if __name__ == "__main__":
    main()