
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os

import os
import urllib.request
import zipfile
from pathlib import Path

def download_eurosat():
    """Download and extract EuroSAT dataset."""
    
    data_dir = './data/eurosat'
    os.makedirs(data_dir, exist_ok=True)
    
    # EuroSAT RGB dataset URL
    url = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip'
    zip_path = os.path.join(data_dir, 'EuroSAT.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading EuroSAT dataset (~90MB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete!")
    
    # Extract
    extract_dir = os.path.join(data_dir, 'EuroSAT')
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✅ Extraction complete!")
    
    return extract_dir

def load_eurosat_features():
    print("========== Extracting EuroSAT Features ==========")
    
    # Download dataset
    dataset_path = download_eurosat()
    
    # Check what's actually in the extracted folder
    print(f"Checking directory structure...")
    
    # EuroSAT extracts to different possible locations, let's find it
    possible_paths = [
        os.path.join(dataset_path, '2750'),           # Expected path
        dataset_path,                                  # Root might be the folder
        os.path.join(dataset_path, 'EuroSAT'),        # Might be nested
        os.path.join('./data/eurosat', '2750'),       # Alternative
    ]
    
    # Find the actual data path
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has subdirectories (class folders)
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if len(subdirs) > 0:
                data_path = path
                print(f"Found EuroSAT data at: {data_path}")
                print(f"Classes found: {subdirs[:3]}... ({len(subdirs)} total)")
                break
    
    if data_path is None:
        # Last resort: search for any folder with class subdirectories
        print("Searching for data recursively...")
        for root, dirs, files in os.walk('./data/eurosat'):
            if len(dirs) >= 10:  # EuroSAT has 10 classes
                # Check if these look like class folders
                sample_dirs = dirs[:3]
                if any(name in ['AnnualCrop', 'Forest', 'Highway'] for name in sample_dirs):
                    data_path = root
                    print(f"Found EuroSAT data at: {data_path}")
                    break
    
    if data_path is None:
        raise FileNotFoundError(
            "Could not find EuroSAT class folders. "
            "Please check ./data/eurosat/ directory structure."
        )
    
    # Transform for ResNet
    transform = transforms.Compose([
        transforms.Resize(224),  # EuroSAT is 64×64, resize to 224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load using ImageFolder
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Total images: {len(full_dataset)}")
    print(f"Classes ({len(full_dataset.classes)}): {full_dataset.classes}")
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    
    # Load ResNet50 (trained on ImageNet - ground photos!)
    print("Loading ImageNet-trained ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    print(f"Extracting features from {len(train_dataset)} training images...")
    
    train_features_list = []
    train_labels_list = []
    
    for i, (images, labels) in enumerate(trainloader):
        with torch.no_grad():
            features = model(images).squeeze()
            
            # Handle single sample case (squeeze might remove batch dim)
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        train_features_list.append(features.cpu().numpy())
        train_labels_list.append(labels.numpy())
        
        if i % 50 == 0:
            print(f"  Train Batch {i+1}/{len(trainloader)}")
    
    train_features = np.concatenate(train_features_list)
    train_labels = np.concatenate(train_labels_list)
    
    print(f"Training features shape: {train_features.shape}")
    
    # Extract test features
    test_features_list = []
    test_labels_list = []
    
    for i, (images, labels) in enumerate(testloader):
        with torch.no_grad():
            features = model(images).squeeze()
            
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        test_features_list.append(features.cpu().numpy())
        test_labels_list.append(labels.numpy())
        
        if i % 50 == 0:
            print(f"  Test Batch {i+1}/{len(testloader)}")
    
    test_features = np.concatenate(test_features_list)
    test_labels = np.concatenate(test_labels_list)
    
    print(f"Test features shape: {test_features.shape}")
    
    # Save features
    np.savez_compressed(
        'eurosat-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels
    )
    
    print("✅ EuroSAT features extracted!")
    print(f"Domain gap: ImageNet (ground photos) → Satellite (aerial views)")
    
def main_features_map():
    if os.path.exists('eurosat-resnet50.npz'):
        print(f"Skipping EuroSAT, features data file already exists")
        data = np.load('eurosat-resnet50.npz')
        print(f"\nEuroSAT File contents:")
        print(f"----Train----")
        print(f"train_features: {data['train_features'].shape}")
        print(f"train_labels: {data['train_labels'].shape}")
        print(f"----Test-----")
        print(f"test_features: {data['test_features'].shape}")
        print(f"test_labels: {data['test_labels'].shape}")
        data.close()
    else:
        print(f"EuroSAT data doesn't exist, generating features now")
        load_eurosat_features()



def main():
    main_features_map()

if __name__ == "__main__":
    main()
