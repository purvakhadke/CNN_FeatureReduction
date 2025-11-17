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


def main():
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

   print("==========ResNet==========")
   model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

   # Remove final FC layer → output becomes 2048-dim feature vectors
   children = list(model.children())
   model = torch.nn.Sequential(*children[:-1])
   model.eval()



if __name__ == "__main__":
   main()
