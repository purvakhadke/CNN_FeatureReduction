
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

# In 0_feature_extractor.py

import scipy.io
import h5py
from PIL import Image

def download_nyu_depth():
    """
    Download NYU Depth V2 dataset.
    
    Note: The full dataset is large (~2.8GB for labeled data)
    We'll use the labeled subset (1,449 images)
    """
    import urllib.request
    
    data_dir = './data/nyu_depth'
    os.makedirs(data_dir, exist_ok=True)
    
    # Download labeled dataset (1,449 RGB-D pairs)
    mat_file = os.path.join(data_dir, 'nyu_depth_v2_labeled.mat')
    
    if not os.path.exists(mat_file):
        print("Downloading NYU Depth V2 labeled dataset (~2.8GB)...")
        print("This may take 10-20 minutes depending on your connection...")
        
        url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
        
        def download_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if count % 10 == 0:  # Update every 50 blocks
                print(f'  Progress: {percent}% ({count * block_size / 1e6:.1f}MB / {total_size / 1e6:.1f}MB)')
        
        urllib.request.urlretrieve(url, mat_file, reporthook=download_progress)
        print("✅ Download complete!")
    else:
        print(f"Dataset already downloaded at {mat_file}")
    
    return data_dir

def load_nyu_depth_features():
    print("========== Extracting NYU Depth V2 Features ==========")
    
    # Download dataset
    data_dir = download_nyu_depth()
    mat_file = os.path.join(data_dir, 'nyu_depth_v2_labeled.mat')
    
    print("Loading depth images from .mat file...")
    print("This might take a few minutes...")
    
    # Load the .mat file
    try:
        with h5py.File(mat_file, 'r') as f:
            depths = np.array(f['depths'])
            labels = np.array(f['labels'])
            images = np.array(f['images'])
            
            print(f"Loaded data shapes:")
            print(f"  Depths: {depths.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  RGB Images: {images.shape}")
            
            # Transpose to correct format
            depths = np.transpose(depths, (0, 2, 1))
            labels = np.transpose(labels, (0, 2, 1))
            images = np.transpose(images, (0, 1, 3, 2))
            
    except Exception as e:
        print(f"h5py failed, trying scipy.io: {e}")
        mat_data = scipy.io.loadmat(mat_file)
        depths = mat_data['depths']
        labels = mat_data['labels']
        images = mat_data['images']
    
    print(f"Successfully loaded {len(depths)} depth images")
    
    # Extract scene labels for classification
    scene_labels = []
    for i in range(len(labels)):
        label_map = labels[i]
        # Get most frequent non-zero label
        unique, counts = np.unique(label_map[label_map > 0], return_counts=True)
        if len(unique) > 0:
            dominant_class = unique[np.argmax(counts)]
        else:
            dominant_class = 0
        scene_labels.append(dominant_class)
    
    scene_labels = np.array(scene_labels)
    
    print(f"Original label distribution (top 20):")
    unique_labels, counts = np.unique(scene_labels, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1][:20]
    for idx in sorted_idx:
        print(f"  Class {unique_labels[idx]}: {counts[idx]} samples")
    
    # Find classes with at least 30 samples (for good train/test split)
    MIN_SAMPLES_PER_CLASS = 30
    label_counts = np.bincount(scene_labels)
    valid_classes = np.where(label_counts >= MIN_SAMPLES_PER_CLASS)[0]
    
    # Take top 10 most common classes with enough samples
    common_classes_counts = [(cls, label_counts[cls]) for cls in valid_classes]
    common_classes_counts.sort(key=lambda x: x[1], reverse=True)
    common_classes = [cls for cls, count in common_classes_counts[:10]]
    
    print(f"\nSelected top 10 classes with ≥{MIN_SAMPLES_PER_CLASS} samples:")
    for cls in common_classes:
        print(f"  Class {cls}: {label_counts[cls]} samples")
    
    # Filter to only keep samples with common classes
    mask = np.isin(scene_labels, common_classes)
    depths_filtered = depths[mask]
    scene_labels_filtered = scene_labels[mask]
    
    # Remap labels to 0-9
    label_mapping = {old: new for new, old in enumerate(common_classes)}
    scene_labels_remapped = np.array([label_mapping[l] for l in scene_labels_filtered])
    
    print(f"\nFiltered to {len(depths_filtered)} images with common classes")
    print(f"Remapped class distribution: {np.bincount(scene_labels_remapped)}")
    
    # Verify all classes have enough samples
    for cls_idx in range(10):
        count = np.sum(scene_labels_remapped == cls_idx)
        print(f"  Class {cls_idx}: {count} samples")
        if count < 10:
            print(f"    ⚠️ Warning: Class {cls_idx} has only {count} samples!")
    
    # Split train/test with stratification
    from sklearn.model_selection import train_test_split
    
    # Use 80/20 split
    train_depths, test_depths, train_labels, test_labels = train_test_split(
        depths_filtered, scene_labels_remapped, 
        test_size=0.2, 
        random_state=42, 
        stratify=scene_labels_remapped
    )
    
    print(f"\nTrain: {len(train_depths)}, Test: {len(test_depths)}")
    print(f"Train class distribution: {np.bincount(train_labels)}")
    print(f"Test class distribution: {np.bincount(test_labels)}")
    
    # Transform for ResNet
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load ResNet50 (trained on RGB photos - NOT depth!)
    print("\nLoading ImageNet-trained ResNet50...")
    print("⚠️  Note: ResNet was trained on RGB photos, NOT depth maps!")
    print("This creates MAXIMUM domain gap - PCA should struggle!")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    print("\nExtracting features from depth images...")
    
    # Process training depth images
    train_features_list = []
    batch_size = 32
    
    for i in range(0, len(train_depths), batch_size):
        batch_depths = train_depths[i:i+batch_size]
        
        batch_tensors = []
        for depth_map in batch_depths:
            # Normalize depth to 0-255 range
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_normalized = (depth_normalized * 255).astype('uint8')
            
            # Convert to PIL Image (grayscale)
            pil_img = Image.fromarray(depth_normalized, mode='L')
            
            # Convert grayscale to RGB (repeat channel 3 times)
            pil_img_rgb = pil_img.convert('RGB')
            
            # Apply transform
            tensor = transform(pil_img_rgb)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        
        with torch.no_grad():
            features = model(batch_tensor).squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        train_features_list.append(features.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"  Train batch {i}/{len(train_depths)}")
    
    train_features = np.concatenate(train_features_list)
    print(f"Train features shape: {train_features.shape}")
    
    # Process test depth images
    test_features_list = []
    
    for i in range(0, len(test_depths), batch_size):
        batch_depths = test_depths[i:i+batch_size]
        
        batch_tensors = []
        for depth_map in batch_depths:
            # Normalize depth to 0-255 range
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_normalized = (depth_normalized * 255).astype('uint8')
            
            # Convert to PIL Image
            pil_img = Image.fromarray(depth_normalized, mode='L')
            pil_img_rgb = pil_img.convert('RGB')
            
            # Apply transform
            tensor = transform(pil_img_rgb)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        
        with torch.no_grad():
            features = model(batch_tensor).squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        test_features_list.append(features.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"  Test batch {i}/{len(test_depths)}")
    
    test_features = np.concatenate(test_features_list)
    print(f"Test features shape: {test_features.shape}")
    
    # Save features
    np.savez_compressed(
        'nyu_depth-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        class_ids=common_classes  # Save original class IDs for reference
    )
    
    print("\n✅ NYU Depth features extracted!")
    print("Domain gap: RGB natural photos → Depth maps (3D geometry)")
    print(f"\nTotal samples: {len(train_features) + len(test_features)}")
    print(f"10 most common indoor object classes selected")




def main_features_map():
    if os.path.exists('nyu_depth-resnet50.npz'):
        print(f"Skipping NYU Depth, features data file already exists")
        data = np.load('nyu_depth-resnet50.npz')
        print(f"\nNYU Depth File contents:")
        print(f"----Train----")
        print(f"train_features: {data['train_features'].shape}")
        print(f"train_labels: {data['train_labels'].shape}")
        print(f"----Test-----")
        print(f"test_features: {data['test_features'].shape}")
        print(f"test_labels: {data['test_labels'].shape}")
        data.close()
    else:
        print(f"NYU Depth data doesn't exist, generating features now")
        load_nyu_depth_features()



def main():
    main_features_map()

if __name__ == "__main__":
    main()
