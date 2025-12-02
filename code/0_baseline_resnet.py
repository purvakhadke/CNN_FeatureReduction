"""
Step A: Baseline - Train ResNet classifier directly on raw CIFAR-10 images
"""
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import time
from config import *

def train_baseline():
    print("\n" + "="*60)
    print("STEP A: Training ResNet50 Baseline on Raw Images")
    print("="*60 + "\n")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    if TRAIN_SAMPLE_SIZE is not None:
        trainset = torch.utils.data.Subset(trainset, range(TRAIN_SAMPLE_SIZE))
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    if TEST_SAMPLE_SIZE is not None:
        testset = torch.utils.data.Subset(testset, range(TEST_SAMPLE_SIZE))

    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )



    # Load pretrained ResNet50 and modify final layer for CIFAR-100
    print("Loading pretrained ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, 100)  # CIFAR-100 has 100 classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\nTraining ResNet50 classifier...")
    start_time = time.time()
    
    for epoch in range(RESNET_EPOCH):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 0:
                print(f"  Epoch [{epoch+1}/{RESNET_EPOCH}], Batch [{i}/{len(trainloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Complete - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"BASELINE RESNET50 TEST ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Save results
    # with open('../results/baseline_results.txt', 'w') as f:
    #     f.write(f"ResNet50 Baseline Results\n")
    #     f.write(f"="*40 + "\n")
    #     f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    #     f.write(f"Training Time: {train_time:.2f} seconds\n")
    
    with open('../results/baseline_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Input_Dim', 'Output_Dim', 'Compression_Ratio', 
                        'Accuracy_%', 'Training_Time_sec'])
        writer.writerow(['ResNet_Baseline', 3072, NUM_CLASSES, '1:1', 
                        accuracy, train_time])

    return accuracy, train_time

def main():
    train_baseline()

if __name__ == "__main__":
    main()