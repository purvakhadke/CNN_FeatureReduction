import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.datasets as datasets
import numpy as np
import os
import urllib.request
from PIL import Image  # ← ADD THIS LINE
from sklearn.model_selection import train_test_split  # ← ADD THIS LINE TOO
# In 0_feature_extractor.py

def download_quickdraw():
    """
    Download Quick, Draw! simplified dataset.
    """
    import urllib.request
    from PIL import Image

    data_dir = './data/quickdraw'
    os.makedirs(data_dir, exist_ok=True)
    
    # Select 10 diverse classes (similar diversity to CIFAR-10)
    classes = [
        'truck', 
        'airplane',          # Vehicle (like plane)
        'apple',            # Fruit/object
        'bicycle',          # Vehicle
        'car',              
        'cat',              # Animal (same as CIFAR-10)
        'dog',              # Animal (same as CIFAR-10)
        'house',            # Structure
        'tree',             # Nature
        'umbrella'          # Object
    ]
    
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    
    print("Downloading Quick, Draw! dataset...")
    print("This will download ~100MB total (10 classes)")
    
    for class_name in classes:
        filepath = os.path.join(data_dir, f'{class_name}.npy')
        
        if os.path.exists(filepath):
            print(f"  {class_name}: Already downloaded ✓")
        else:
            url = f'{base_url}{class_name}.npy'
            print(f"  Downloading {class_name}...", end=' ')
            try:
                urllib.request.urlretrieve(url, filepath)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print('✅ Quick, Draw! dataset ready!')
    return data_dir, classes

def load_quickdraw_features():
    print("========== Extracting Quick, Draw! Features ==========")
    
    # Download dataset
    data_dir, classes = download_quickdraw()
    
    print(f"\nLoading {len(classes)} classes of hand-drawn sketches...")
    
    # Load and combine all classes
    all_images = []
    all_labels = []
    
    SAMPLES_PER_CLASS = 5000  # Use 5000 samples per class (50,000 total)
    
    for idx, class_name in enumerate(classes):
      filepath = os.path.join(data_dir, f'{class_name}.npy')
      
      if not os.path.exists(filepath):
         print(f"  ⚠️ {class_name} not found, skipping...")
         continue
      
      try:
         # Load with memory mapping (safer for large files)
         images = np.load(filepath, mmap_mode='r')
         
         # Check if it's the expected format (N, 784)
         if images.ndim != 2 or images.shape[1] != 784:
               print(f"  ⚠️ {class_name}: Unexpected shape {images.shape}, skipping...")
               continue
         
         print(f'  {class_name}: {len(images)} available → ', end='')
         
         # Take first SAMPLES_PER_CLASS and copy to memory
         num_to_take = min(SAMPLES_PER_CLASS, len(images))
         images_subset = np.array(images[:num_to_take])
         
         all_images.append(images_subset)
         all_labels.append(np.full(num_to_take, idx))
         
         print(f'Using {num_to_take} ✓')
         
      except Exception as e:
         print(f'  ✗ {class_name}: Error - {e}')
         continue
    # Combine all data
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    
    print(f'\nTotal sketches: {len(all_images):,}')
    print(f'Image shape: 28×28 grayscale')
    print(f'Classes: {len(classes)}')
    
    # Reshape from flat (784,) to 2D (28, 28)
    all_images = all_images.reshape(-1, 28, 28)
    
    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=all_labels
    )
    
    print(f'\nSplit: {len(train_images):,} train, {len(test_images):,} test')
    
    # Transform for ResNet (28×28 → 224×224, grayscale → RGB)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load ResNet50 (trained on natural PHOTOS, not SKETCHES!)
    print("\nLoading ImageNet-trained ResNet50...")
    print("⚠️  CRITICAL: ResNet was trained on PHOTOS, not SKETCHES!")
    print("This creates MAXIMUM domain gap:")
    print("  - Photorealistic images → Abstract line drawings")
    print("  - Color/texture → Black & white lines")
    print("  - 3D rendered → 2D simplified")
    print("\nExpected: PCA will struggle significantly!")
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    print("\nExtracting features from hand-drawn sketches...")
    
    # Process training sketches
    train_features_list = []
    batch_size = 128  # Larger batch since images are small
    
    for i in range(0, len(train_images), batch_size):
        batch_images = train_images[i:i+batch_size]
        
        # Convert to PIL and transform
        batch_tensors = []
        for img in batch_images:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img.astype('uint8'), mode='L')  # 'L' = grayscale
            
            # Convert grayscale to RGB (ResNet expects 3 channels)
            pil_img_rgb = pil_img.convert('RGB')
            
            # Apply transform (resize to 224×224, normalize)
            tensor = transform(pil_img_rgb)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        
        # Extract features
        with torch.no_grad():
            features = model(batch_tensor).squeeze()
            
            # Handle single sample case
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        train_features_list.append(features.cpu().numpy())
        
        # Progress update every 5000 images
        if i % 5000 == 0:
            print(f'  Train: {i:,}/{len(train_images):,} processed...')
    
    train_features = np.concatenate(train_features_list)
    print(f'✓ Train features shape: {train_features.shape}')
    
    # Process test sketches
    test_features_list = []
    
    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i+batch_size]
        
        batch_tensors = []
        for img in batch_images:
            pil_img = Image.fromarray(img.astype('uint8'), mode='L')
            pil_img_rgb = pil_img.convert('RGB')
            tensor = transform(pil_img_rgb)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        
        with torch.no_grad():
            features = model(batch_tensor).squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
        
        test_features_list.append(features.cpu().numpy())
        
        if i % 5000 == 0:
            print(f'  Test: {i:,}/{len(test_images):,} processed...')
    
    test_features = np.concatenate(test_features_list)
    print(f'✓ Test features shape: {test_features.shape}')
    
    # Save features
    np.savez_compressed(
        'quickdraw-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels
    )
    
    print('\n' + '='*70)
    print('✅ Quick, Draw! features extracted!')
    print('='*70)
    print('Domain gap: Photorealistic images → Hand-drawn sketches')
    print('This is the MAXIMUM possible visual domain gap!')
    print(f'Total samples: {len(train_features) + len(test_features):,}')
    print(f'Classes: {classes}')
    print('\nExpected Results:')
    print('  PCA:         45-52% (struggles with photo→sketch gap)')
    print('  Autoencoder: 65-72% (+20%)')
    print('  Transformer: 68-75% (+23%) ✅ MASSIVE ADVANTAGE')
    print('='*70)

def main_features_map():
    # Check which features to extract
   
   if os.path.exists('quickdraw-resnet50.npz'):
      print(f"Quick Draw features already exist")
      data = np.load('quickdraw-resnet50.npz')
      print(f"\nQuick Draw File contents:")
      print(f"----Train----")
      print(f"train_features: {data['train_features'].shape}")
      print(f"train_labels: {data['train_labels'].shape}")
      print(f"----Test-----")
      print(f"test_features: {data['test_features'].shape}")
      print(f"test_labels: {data['test_labels'].shape}")
      data.close()
   else:
      load_quickdraw_features()



def main():
    main_features_map()

if __name__ == "__main__":
    main()
