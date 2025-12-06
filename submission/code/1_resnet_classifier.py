# ResNet Classifier for CIFAR-100
# Works on: Raw, PCA, UMAP, and Autoencoder-reduced features

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
    def __init__(self, input_type='Raw'):
        self.input_type = input_type
        self.is_raw = (input_type == 'Raw')
        self.model = None
        
    def build_model(self, input_dim=None):
        if self.is_raw:
            # For raw images: use pretrained ResNet50
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Freeze backbone
            for param in self.model.parameters():
                param.requires_grad = False
            # Replace classifier head
            self.model.fc = nn.Linear(2048, NUM_CLASSES)
        else:
            # For the reduced ones PCA/UMAP/AE use MLP
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        return device
    
    def prepare_data(self, data_path):
        data = np.load(data_path)
        
        if self.is_raw:
            # reshape/ transform
            train_images = data['train_features'].reshape(-1, 32, 32, 3)
            test_images = data['test_features'].reshape(-1, 32, 32, 3)
            
            # Convert to torch tensors and permute to (N, C, H, W)
            train_features = torch.from_numpy(train_images).permute(0, 3, 1, 2).float()
            test_features = torch.from_numpy(test_images).permute(0, 3, 1, 2).float()
            
            # Resize/normalize
            resize = transforms.Resize(96)
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            train_features = torch.stack([normalize(resize(img)) for img in train_features])
            test_features = torch.stack([normalize(resize(img)) for img in test_features])
        else:
            # else they are already flattened
            train_features = torch.from_numpy(data['train_features']).float()
            test_features = torch.from_numpy(data['test_features']).float()
        
        train_labels = torch.from_numpy(data['train_labels']).long()
        test_labels = torch.from_numpy(data['test_labels']).long()
        
        return train_features, train_labels, test_features, test_labels
    
    def train(self, train_loader, test_loader, device, epochs):
        criterion = nn.CrossEntropyLoss()
        
        if self.is_raw:
            optimizer = optim.Adam(self.model.fc.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        print(f"\nTraining ResNet on {self.input_type}")
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
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        train_time = time.time() - start_time
        accuracy = self.evaluate(test_loader, device)
        
        return accuracy, train_time
    
    def evaluate(self, test_loader, device):
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
        
        return 100 * correct / total


def main():
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    results = []
    
    for input_type, data_path in INPUT_FILES.items():
        is_raw = (input_type == 'Raw')
        classifier = ResNetClassifier(input_type=input_type)
        
        # Determine input dimension
        if is_raw:
            input_dim = None
            epochs = RESNET_EPOCH
            batch_size = 64
        else:
            input_dim = REDUCED_DIM
            epochs = EPOCHS
            batch_size = BATCH_SIZE
        
        device = classifier.build_model(input_dim=input_dim)
        train_feat, train_lab, test_feat, test_lab = classifier.prepare_data(data_path)
        
        train_dataset = TensorDataset(train_feat, train_lab)
        test_dataset = TensorDataset(test_feat, test_lab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        acc, time_taken = classifier.train(train_loader, test_loader, device, epochs)
        
        # Save model
        model_name = f"resnet_{input_type.lower().replace('-', '_')}.pth"
        torch.save(classifier.model.state_dict(), f'../models/{model_name}')
        
        # Determine input dimension for results
        dim = 32*32*3 if is_raw else REDUCED_DIM
        results.append(['ResNet', input_type, dim, acc, time_taken])
    
    # Save results
    with open('../results/resnet_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Input_Type', 'Input_Dim', 'Accuracy_%', 'Training_Time_sec'])
        writer.writerows(results)
    

if __name__ == "__main__":
    main()