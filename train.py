import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mnist_model import MNISTNet
from datetime import datetime

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced data augmentation with less aggressive transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(5),  # Reduced rotation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Reduced translation
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size
    
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)  # Increased learning rate
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader)//4,  # Shorter cycle length
        T_mult=1,
        eta_min=1e-4
    )
    
    # Training for exactly 1 epoch
    print("Starting training for 1 epoch...")
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Mixed precision for faster training
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(data)
            loss = criterion(output, target)
        
        loss.backward()
        # Gradient clipping
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
    
    # Final training accuracy
    final_acc = 100 * correct / total
    print(f'\nTraining completed. Final accuracy: {final_acc:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/mnist_model_{timestamp}_acc{final_acc:.1f}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train() 