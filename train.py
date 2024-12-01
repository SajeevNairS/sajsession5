import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mnist_model import MNISTNet
from datetime import datetime

def train():
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # MNIST-specific augmentation
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    print("Starting training for 1 epoch...")
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            current_acc = 100 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {current_acc:.2f}%')
    
    final_acc = 100 * correct / total
    print(f'\nTraining completed. Final accuracy: {final_acc:.2f}%')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/mnist_model_{timestamp}_acc{final_acc:.1f}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train() 