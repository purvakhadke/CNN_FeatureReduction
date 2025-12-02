"""
ResNet Classifier for CIFAR-100
Can work on:
1. Raw images (3072-D reshaped to 32x32x3)
2. PCA features (512-D)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import time
import csv
import os
from config import *


class ResNetClassifier:
    def __init__(self, use_pca=False):
        self.use_pca = use_pca
        self.model = None
        
    def build_model(self, input_dim=None):
        """Build ResNet model"""
        if self.use_pca:
            # For PCA input: simple MLP that mimics ResNet structure
            self.model = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, NUM_CLASSES)
            )
        else:
            # For raw images: use pretrained ResNet50
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Freeze backbone
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Replace classifier head
            self.model.fc = nn.Linear(2048, NUM_CLASSES)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        return device
    
    def prepare_data(self, data_path):
        """Load and prepare data"""
        data = np.load(data_path)
        
        if self.use_pca:
            # PCA features - already flattened
            train_features = torch.from_numpy(data['train_features']).float()
            test_features = torch.from_numpy(data['test_features']).float()
        else:
            # Raw images - reshape and apply transforms
            train_images = data['train_features'].reshape(-1, 32, 32, 3)
            test_images = data['test_features'].reshape(-1, 32, 32, 3)
            
            # Convert to torch tensors and permute to (N, C, H, W)
            train_features = torch.from_numpy(train_images).permute(0, 3, 1, 2).float()
            test_features = torch.from_numpy(test_images).permute(0, 3, 1, 2).float()
            
            # Resize to RESNET_IMAGE_SIZE and normalize
            resize = transforms.Resize(RESNET_IMAGE_SIZE)
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            train_features = torch.stack([normalize(resize(img)) for img in train_features])
            test_features = torch.stack([normalize(resize(img)) for img in test_features])
        
        train_labels = torch.from_numpy(data['train_labels']).long()
        test_labels = torch.from_numpy(data['test_labels']).long()
        
        return train_features, train_labels, test_features, test_labels
    
    def train(self, train_loader, test_loader, device, epochs):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        
        if self.use_pca:
            optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        else:
            # Only train the classifier head for ResNet
            optimizer = optim.Adam(self.model.fc.parameters(), lr=LEARNING_RATE)
        
        print(f"\nTraining ResNet ({'with PCA' if self.use_pca else 'on raw images'})...")
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if i % 50 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        train_time = time.time() - start_time
        
        # Evaluate
        accuracy = self.evaluate(test_loader, device)
        
        return accuracy, train_time
    
    def evaluate(self, test_loader, device):
        """Evaluate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy


def main():
    os.makedirs('../results', exist_ok=True)
    results = []
    
    # Test 1: ResNet on raw images
    print("\n" + "="*60)
    print("TEST 1: ResNet on Raw Images (3072-D)")
    print("="*60)
    
    classifier = ResNetClassifier(use_pca=False)
    device = classifier.build_model()
    train_feat, train_lab, test_feat, test_lab = classifier.prepare_data('../data/cifar100_raw.npz')
    
    train_dataset = TensorDataset(train_feat, train_lab)
    test_dataset = TensorDataset(test_feat, test_lab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    acc, time_taken = classifier.train(train_loader, test_loader, device, RESNET_EPOCH)
    print(f"\n✅ ResNet (Raw) - Accuracy: {acc:.2f}%, Time: {time_taken:.2f}s")
    results.append(['ResNet', 'Raw', FLATTENED_DIM, acc, time_taken])
    
    # Test 2: ResNet on PCA features
    print("\n" + "="*60)
    print(f"TEST 2: ResNet on PCA Features ({PCA_COMPONENTS}-D)")
    print("="*60)
    
    classifier = ResNetClassifier(use_pca=True)
    device = classifier.build_model(input_dim=PCA_COMPONENTS)
    train_feat, train_lab, test_feat, test_lab = classifier.prepare_data(
        f'../data/cifar100_pca{PCA_COMPONENTS}.npz'
    )
    
    train_dataset = TensorDataset(train_feat, train_lab)
    test_dataset = TensorDataset(test_feat, test_lab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    acc, time_taken = classifier.train(train_loader, test_loader, device, EPOCHS)
    print(f"\n✅ ResNet (PCA) - Accuracy: {acc:.2f}%, Time: {time_taken:.2f}s")
    results.append(['ResNet', f'PCA-{PCA_COMPONENTS}', PCA_COMPONENTS, acc, time_taken])
    
    # Save results
    with open('../results/resnet_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Input_Type', 'Input_Dim', 'Accuracy_%', 'Training_Time_sec'])
        writer.writerows(results)
    
    print(f"\n✅ Results saved to '../results/resnet_results.csv'\n")


if __name__ == "__main__":
    main()