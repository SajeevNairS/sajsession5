import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
import torchviz
from torchviz import make_dot
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mnist_model import MNISTNet
from datetime import datetime
from utils.augmentation_viz import visualize_augmentations

# Optional imports
try:
    import torchviz
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False

def train():
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # MNIST-specific augmentation
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        #transforms.RandomAffine(degrees=5,translate=(0.05, 0.05),scale=(0.95, 1.05)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4,              # Reduced from 128 to 32
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001,                   # Slightly increased learning rate
        momentum=0.9
    )
    #optimizer = optim.AdamW(
    #    model.parameters(),
    #    lr=0.005,                   # Slightly increased learning rate
    #    weight_decay=0.01,
    #    amsgrad=True
    #)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,              # Adjusted max_lr
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,             # Shorter warmup due to smaller batches
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
        
        # Removed CUDA mixed precision since we're using CPU
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

def visualize_model():
    # Create model instance
    model = MNISTNet()
    
    # Check if visualization should be skipped
    SKIP_VIZ = os.environ.get('SKIP_VIZ', '0') == '1'
    
    # Clean up old visualizations
    import shutil
    if os.path.exists('visualizations'):
        shutil.rmtree('visualizations')
    os.makedirs('visualizations')
    os.makedirs('visualizations/augmentations', exist_ok=True)
    
    # Get a sample image for augmentation visualization
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sample_image = train_dataset[0][0]
    
    # Visualize augmentations if not skipped
    if not SKIP_VIZ:
        print("\nGenerating augmentation visualizations...")
        visualize_augmentations(sample_image, num_samples=5)
        print("Augmentation visualizations saved in visualizations/augmentations/")
    
    # Create model architecture visualization if not skipped
    if TORCHVIZ_AVAILABLE and not SKIP_VIZ:
        try:
            x = torch.randn(1, 1, 28, 28)
            y = model(x)
            dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
            dot.render('visualizations/model_architecture', format='png', cleanup=True)
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
    
    # Calculate and display total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("\nModel Architecture:")
    print("==================")
    print(f"Total Parameters: {total_params:,}")
    
    # Create model architecture visualization
    with open('visualizations/model_architecture.txt', 'w') as f:
        f.write("MNIST Model Architecture\n")
        f.write("======================\n\n")
        
        # Input
        f.write("Input Layer: (1, 28, 28)\n")
        f.write("↓\n")
        
        # Dynamically document each layer
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                f.write(f"\n{name}: {module.kernel_size[0]}x{module.kernel_size[1]}, ")
                f.write(f"{module.in_channels}->{module.out_channels} channels\n")
                params = sum(p.numel() for p in module.parameters())
                f.write(f"Parameters: {params:,}\n")
                f.write("↓\n")
                
            elif isinstance(module, nn.MaxPool2d):
                f.write(f"{name}: kernel={module.kernel_size}, stride={module.stride}\n")
                f.write("↓\n")
                
            elif isinstance(module, nn.Linear):
                f.write(f"\n{name}: {module.in_features} -> {module.out_features}\n")
                params = sum(p.numel() for p in module.parameters())
                f.write(f"Parameters: {params:,}\n")
                f.write("↓\n")
        
        # Total parameters
        f.write(f"\nTotal Parameters: {total_params:,}\n")
    
    # Rest of the visualization code for features...

def visualize_features(model, sample_data, sample_idx, label):
    # Get all feature extracting layers
    feature_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            feature_layers.append((name, module))
    
    # Forward pass with hooks
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    for name, layer in feature_layers:
        hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot input image
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_data.squeeze(), cmap='gray')
    plt.title(f'Input Image {sample_idx+1} (Label: {label})')
    plt.axis('off')
    plt.savefig(f'visualizations/input_{sample_idx+1}.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Plot feature maps for each layer
    for layer_name, features in activations.items():
        features = features[0]
        n_features = features.shape[0]
        
        rows = int(np.ceil(np.sqrt(n_features)))
        fig, axes = plt.subplots(rows, rows, figsize=(12, 12))
        fig.suptitle(f'{layer_name} Feature Maps\nInput {sample_idx+1} (Label: {label})', size=16)
        
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(n_features):
            ax = axes[idx//rows, idx%rows]
            ax.imshow(features[idx].numpy(), cmap='viridis')
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_features, rows*rows):
            axes[idx//rows, idx%rows].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{layer_name}_features_{sample_idx+1}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()

if __name__ == "__main__":
    visualize_model()
    train() 