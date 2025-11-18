'''
Lets put relevant resources/citations here for
- maybe to be cited in our paper
- or just links to documentation to topics you think should be noted 

Original Paper for ResNet: https://arxiv.org/abs/1512.03385
(using ResNet) How to use pre-trained models in Python: https://docs.pytorch.org/vision/main/models.html


===================================================================
Note: need to follow up w TA if PCA or UMAP is required, might have been a typo that he included autoencoders in the next line
Typo?-> "Extract CNN features then apply PCA, UMAP, or autoencoders."
- regardless , it shouldnt be too hard to implement

4. Reduce deep CNN feature representations for visualization or classification.
   - Extract CNN features then apply PCA, UMAP, or autoencoders.
   - Train a classifier on reduced space. 
   - Compare Autoencoder vs Transformer-based reduction. 
   - Analyze feature interpretability in reduced spaces.
===================================================================

1. Extract CNN Features (ResNet)
   - Load CIFAR-10 images (3072 dim (32x32x3))
   - Pass through ResNet50 (a pretrained model)
   - Output is 2048 dim feature vectors
   - Save as 'cifar10-resnet50.tar.gz'
   - Load to repo for AutoEncoder and Transformer reductions to use

2. Autoencoder Reduction
   - Load 'cifar10-resnet50.npz' (2048 dim)
   - research TBD to figure out exact steps of autoencoder training

3. Transformer Reduction
   - Load 'cifar10-resnet50.npz' (2,048-dim features)
   - research TBD to figure out exact steps of transformer training

4. Compare Results
   - Compare accuracies: Autoencoder vs Transformer
   - We need to have enough info to answer all RQs

'''

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os

# get ResNet features layer
def load_resnet_features():

   # CIFAR-10 has 60000 32x32 color images across 10 classes
   classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   # ResNet50 expects 224x224 images
   transform = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform
   )
   trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=64, shuffle=False, num_workers=2
   )

   testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform
   )
   testloader = torch.utils.data.DataLoader(
      testset, batch_size=64, shuffle=False, num_workers=2
   )

   # Load ResNet50 and remove final classification layer
   print("==========ResNet==========")
   model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

   '''
   used llm to help understand/visualize this idea
      Full ResNet50 architecture:
      
      Input (224x224x3)
         ↓
      [Convolutional layers + Residual blocks]  ← Extract visual features
         ↓
      Features (2048-dim)  ← This is what we want!
         ↓
      [Fully Connected Layer]  ← Maps 2048 → 1000 classes (ImageNet)
         ↓
      Output (1000 classes for ImageNet)  ← We don't need this!
   '''
   children = list(model.children())
   model = torch.nn.Sequential(*children[:-1])
   
   model.eval()

   print(f"Length of trainloader: {len(trainloader)}")   
   train_features_list = []
   train_labels_list = []

   for i, (images, labels) in enumerate(trainloader):
      
      # since we're not training, we don't need to calculate the gradients for our outputs (alot is same as the tutorial)
      with torch.no_grad():
         features = model(images).squeeze()
      
      train_features_list.append(features.cpu().numpy())
      train_labels_list.append(labels.numpy())
      
      # Progress logging
      if i % 100 == 0:
         print(f"Doing Batch {i+1}/{len(trainloader)}")

   train_features = np.concatenate(train_features_list)
   train_labels = np.concatenate(train_labels_list)

   print(f"Training shape: {train_features.shape}")

   print(f"Length of testloader: {len(testloader)}")

   test_features_list = []
   test_labels_list = []

   for i, (images, labels) in enumerate(testloader):
      
      # Since we're not training, we don't need to calculate the gradients
      with torch.no_grad():
         features = model(images).squeeze()
      
      test_features_list.append(features.cpu().numpy())
      test_labels_list.append(labels.numpy())
      
      
      if i % 100 == 0:
         print(f"  Batch {i}/{len(testloader)}")

   test_features = np.concatenate(test_features_list)
   test_labels = np.concatenate(test_labels_list)
   print(f"Test shape: {test_features.shape}")

   # download features into tar.dz file kike the HW
   filename = 'cifar10-resnet50.tar.gz'
   np.savez_compressed(
      filename.replace('.tar.gz', ''),
      train_features=train_features,
      train_labels=train_labels,
      test_features=test_features,
      test_labels=test_labels
   )

def main():
   
   if os.path.exists('cifar10-resnet50.npz'):
      print(f"Skipping ResNet, features data file already exists")
   else:
      print(f"ResNet data doesnt exist, generateing features data now")
      load_resnet_features()


if __name__ == "__main__":
   main()
