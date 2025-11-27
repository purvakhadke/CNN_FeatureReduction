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
from pycocotools.coco import COCO
import urllib.request
import zipfile

def download_coco():
    """Download COCO 2017 TRAINING dataset (more images than validation)."""
    data_dir = './data/coco'
    os.makedirs(data_dir, exist_ok=True)
    
    # Use TRAINING set (118K images) instead of validation (5K images)
    train_img_url = 'http://images.cocodataset.org/zips/train2017.zip'
    train_ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    
    train_img_zip = os.path.join(data_dir, 'train2017.zip')
    train_ann_zip = os.path.join(data_dir, 'annotations_trainval2017.zip')
    
    print("="*70)
    print("Downloading COCO 2017 TRAINING Dataset")
    print("="*70)
    print("⚠️  Training set is ~18GB (larger than validation ~1GB)")
    print("⚠️  But has MUCH more images per category")
    print()
    
    # Download training images
    if not os.path.exists(os.path.join(data_dir, 'train2017')):
        print("Downloading training images (~18GB)...")
        print("This will take a while...")
        if not os.path.exists(train_img_zip):
            urllib.request.urlretrieve(train_img_url, train_img_zip)
        print("Extracting...")
        with zipfile.ZipFile(train_img_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✓ Done")
    else:
        print("✓ Training images already exist")
    
    # Download annotations
    if not os.path.exists(os.path.join(data_dir, 'annotations')):
        print("Downloading annotations (~500MB)...")
        if not os.path.exists(train_ann_zip):
            urllib.request.urlretrieve(train_ann_url, train_ann_zip)
        print("Extracting...")
        with zipfile.ZipFile(train_ann_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✓ Done")
    else:
        print("✓ Annotations already exist")
    
    print("✅ COCO dataset ready!\n")
    return data_dir

def load_coco_relaxed():
    """
    Load COCO with RELAXED filtering - gets more images.
    Each image labeled by its MOST PROMINENT object.
    """
    print("="*70)
    print("Extracting COCO Features (RELAXED filtering)")
    print("="*70)
    
    data_dir = download_coco()
    ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    coco = COCO(ann_file)
    
    # Select categories with MOST images available
    COMMON_CATEGORIES = [
        'person',      # ~500+ available
        'car',         # ~500+ available  
        'chair',       # ~500+ available
        'dining table', # ~500+ available
        'motorcycle',  # ~500+ available
        'bus',         # ~500+ available
        'truck',       # ~500+ available
        'bicycle',     # ~500+ available
        'airplane',    # ~500+ available
        'train'        # ~500+ available
    ]
    
    cats = coco.loadCats(coco.getCatIds())
    selected_cat_ids = []
    selected_cat_names = []
    for cat in cats:
        if cat['name'] in COMMON_CATEGORIES:
            selected_cat_ids.append(cat['id'])
            selected_cat_names.append(cat['name'])
    
    print(f"Selected {len(selected_cat_names)} categories: {selected_cat_names}\n")
    
    # RELAXED FILTERING: Just need the category to be PRESENT
    all_images = []
    all_labels = []
    
    # MAX_IMAGES_PER_CLASS = 500  # Target per class
    MAX_IMAGES_PER_CLASS = 6000  # Target per class
    
    for idx, cat_id in enumerate(selected_cat_ids):
        img_ids = coco.getImgIds(catIds=[cat_id])
        
        count = 0
        for img_id in img_ids:
            if count >= MAX_IMAGES_PER_CLASS:
                break
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
            anns = coco.loadAnns(ann_ids)
            
            # RELAXED: Just need at least one instance of the category
            if len(anns) > 0:
                img_info = coco.loadImgs(img_id)[0]
                all_images.append(img_info)
                all_labels.append(idx)
                count += 1
        
        print(f"  {selected_cat_names[idx]}: {count} images")
    
    print(f"\nTotal images collected: {len(all_images)}")
    
    # Load ResNet50
    print("\nLoading ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\nExtracting features...")
    
    features_list = []
    labels_list = []
    
    batch_size = 32
    img_dir = os.path.join(data_dir, 'train2017')
    
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i+batch_size]
        batch_labels = all_labels[i:i+batch_size]
        
        batch_tensors = []
        batch_valid_labels = []
        
        for img_info, label in zip(batch_images, batch_labels):
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img)
                batch_tensors.append(tensor)
                batch_valid_labels.append(label)
            except Exception as e:
                continue
        
        if len(batch_tensors) == 0:
            continue
        
        batch_tensor = torch.stack(batch_tensors)
        
        with torch.no_grad():
            batch_features = model(batch_tensor).squeeze()
            if batch_features.dim() == 1:
                batch_features = batch_features.unsqueeze(0)
        
        features_list.append(batch_features.cpu().numpy())
        labels_list.extend(batch_valid_labels)
        
        if i % 500 == 0:
            print(f"  Processed {i}/{len(all_images)} images...")
    
    all_features = np.concatenate(features_list)
    all_labels = np.array(labels_list)
    
    print(f"\n✓ Features: {all_features.shape}")
    print(f"✓ Labels: {all_labels.shape}")
    
    # Split 80/20
    train_features, test_features, train_labels, test_labels = train_test_split(
        all_features, all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    
    print(f"\nTrain: {len(train_features)} samples")
    print(f"Test:  {len(test_features)} samples")
    
    # Save
    np.savez_compressed(
        'coco-resnet50',
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        class_names=selected_cat_names
    )
    
    print("\n" + "="*70)
    print("✅ COCO features extracted!")
    print("="*70)
    print(f"Total: {len(train_features) + len(test_features)} images")
    print(f"Classes: {len(selected_cat_names)}")
    print("="*70)

def main():
    if os.path.exists('coco-resnet50.npz'):
        print("COCO features already exist. Delete coco-resnet50.npz to re-extract.")
        data = np.load('coco-resnet50.npz', allow_pickle=True)
        print(f"\nExisting dataset:")
        print(f"  Train: {data['train_features'].shape}")
        print(f"  Test:  {data['test_features'].shape}")
        print(f"  Classes: {list(data['class_names'])}")
        data.close()
    else:
        load_coco_relaxed()

if __name__ == "__main__":
    main()