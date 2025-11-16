import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import torch
import torchvision
import torchvision.transforms as transforms

'''
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

   1. Extract CNN features (w ResNet)
      - ResNet extracts CNN features (reducing 32x32x3 dim to 2048dim)
         - using ResNet50 (already trained) to get the deep features from CIRFAR-10 pics
         - save the features as cifar10-resnet50.tar.gz
      
   2. Apply Autoencoder based reduction

   3. Apply Transformer based reduction

   1. Extract CNN Features
      - Load CIFAR-10 images (3072 dim (32x32x3))
      - Pass through ResNet50 (a pretrained model)
      - Output is 2048 dim feature vecotrs
      - Save as 'cifar10-resnet50.tar.gz'
      - Load to repo for AutoEncoder and Transformer reductions to use

   2. Autoencoder Reduction
      - Load 'cifar10-resnet50.tar.gznpz' (2048 dim)
      - research TBD to figure out exact steps of autoencoder training

   3. Transformer Reduction (Person 3)
      - Load 'cifar10-resnet50.tar.gz' (2,048-dim features)
      - research TBD to figure out exact steps of transformer training

   4. Compare Results
      - Compare accuracies: Autoencoder vs Transformer
      - We need to have enough info to answer all RQs

'''

if __name__ == '__main__':

    # CIFAR-10 dataset has of 60000 32x32 color images per class
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')