import sys
import os
import torch
import pytest
from torchvision import datasets, transforms
import base64
from pathlib import Path
import glob
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
import torchvision.transforms.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mnist_model import MNISTNet
from train import train
from utils.augmentation_viz import visualize_augmentations

def count_parameters(model):
    """Count trainable parameters in the model"""
    total_params = 0
    layer_params = {}
    
    # Count parameters for each layer
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            layer_params[name] = param_count
            total_params += param_count
    
    # Print detailed parameter count
    print("\nParameter Count Details:")
    print("=" * 40)
    for name, count in layer_params.items():
        print(f"{name}: {count:,} parameters")
    print("=" * 40)
    print(f"Total Parameters: {total_params:,}")
    
    return total_params

def generate_test_summary(param_count, accuracy, class_accuracies):
    """Generate a markdown summary of test results"""
    summary = []
    summary.append("# MNIST Model Test Results\n")
    
    # Data Augmentation Section
    summary.append("## Data Augmentation Examples\n")
    if os.path.exists('visualizations/augmentations'):
        summary.append("### Sample Augmentations\n")
        
        # Add links to per-digit folders
        for digit_folder in sorted(os.listdir('visualizations/augmentations')):
            if digit_folder.startswith('digit_'):
                digit = digit_folder.split('_')[1]
                summary.append(f"#### Digit {digit}\n")
                
                # Original image with GitHub path
                orig_path = f"visualizations/augmentations/{digit_folder}/original.png"
                if os.path.exists(orig_path):
                    summary.append("Original:")
                    summary.append(f"![Original](https://github.com/SajeevNairS/sajsession5/raw/main/{orig_path})\n")
                
                # Grid images with GitHub paths
                for aug_type in ['rotation', 'affine', 'combined']:
                    grid_path = f"visualizations/augmentations/{digit_folder}/{aug_type}_grid.png"
                    if os.path.exists(grid_path):
                        summary.append(f"{aug_type.title()} Augmentations:")
                        summary.append(f"![{aug_type}](https://github.com/SajeevNairS/sajsession5/raw/main/{grid_path})\n")
    
    # Parameter Count Section
    summary.append("## Parameter Count Check")
    summary.append(f"- **Target:** < 25,000 parameters")
    summary.append(f"- **Actual:** {param_count:,} parameters")
    summary.append(f"- **Margin:** {25000 - param_count:,} parameters remaining")
    status = "✅ PASSED" if param_count < 25000 else "❌ FAILED"
    summary.append(f"- **Status:** {status}\n")
    
    # Accuracy Section
    summary.append("## Accuracy Check")
    summary.append(f"- **Target:** ≥ 95.00%")
    summary.append(f"- **Actual:** {accuracy:.2f}%")
    summary.append(f"- **Margin:** {accuracy - 95.0:+.2f}%")
    status = "✅ PASSED" if accuracy >= 95.0 else "❌ FAILED"
    summary.append(f"- **Status:** {status}\n")
    
    # Per-Class Accuracy
    summary.append("\n## Per-Class Performance")
    summary.append("| Digit | Accuracy | Status |")
    summary.append("|-------|----------|---------|")
    for digit, acc in class_accuracies.items():
        status = "✅" if acc >= 95.0 else "⚠️"
        summary.append(f"| {digit} | {acc:.2f}% | {status} |")
    summary.append("")
    
    # Overall Status
    summary.append("## Overall Status")
    if accuracy >= 95.0 and param_count < 25000:
        summary.append("✅ **ALL CHECKS PASSED**")
    else:
        summary.append("❌ **SOME CHECKS FAILED**")
    
    # Save summary to file
    with open('test-summary.md', 'w') as f:
        f.write('\n'.join(summary))
    
    return '\n'.join(summary)

def test_model():
    """Run model verification tests"""
    # Force CPU
    device = torch.device('cpu')
    torch.set_num_threads(1)  # Use single thread for consistency
    
    print("\nRunning Model Verification Tests")
    print("================================")
    print(f"Using device: {device}")
    
    # Ensure directories exist
    os.makedirs('visualizations/augmentations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate augmentation samples first
    print("\nGenerating augmentation samples...")
    generate_augmentation_samples()
    
    # Verify augmentations were generated
    if not os.path.exists('visualizations/augmentations/README.md'):
        raise FileNotFoundError("Augmentation README.md not generated!")
    
    png_files = list(Path('visualizations/augmentations').rglob('*.png'))
    if not png_files:
        raise FileNotFoundError("No augmentation images were generated!")
    
    print(f"Generated {len(png_files)} augmentation images")
    
    # Parameter count test
    model = MNISTNet()
    param_count = count_parameters(model)
    
    # Load trained model
    model_path = os.environ.get('TRAINED_MODEL_PATH')
    if not model_path:
        print("No model path provided, searching in models directory...")
        model_files = glob.glob('models/*.pth')
        if not model_files:
            print("No trained model found, training new model...")
            model_path = train()
        else:
            model_path = max(model_files, key=os.path.getctime)
            print(f"Found existing model: {model_path}")
    
    print(f"Using model: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test dataset setup
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform, download=True)
    test_subset_size = 2000
    indices = torch.randperm(len(test_dataset))[:test_subset_size]
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=500,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate accuracy
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate accuracies
    accuracy = 100 * correct / total
    class_accuracies = {
        digit: 100 * correct / total 
        for digit, (correct, total) in enumerate(zip(class_correct.values(), class_total.values()))
        if total > 0
    }
    
    # Generate and print summary
    summary = generate_test_summary(param_count, accuracy, class_accuracies)
    print(summary)
    
    # Assertions
    assert param_count < 25000, f"Parameter count {param_count:,} exceeds limit of 25,000"
    assert accuracy >= 95.0, f"Accuracy {accuracy:.2f}% below target of 95%"

def generate_augmentation_samples():
    """Generate augmentation samples for visualization"""
    print("\nGenerating augmentation samples...")
    
    # Create main augmentations directory
    augmentation_dir = 'visualizations/augmentations'
    os.makedirs(augmentation_dir, exist_ok=True)
    
    # Create main README
    readme_path = os.path.join(augmentation_dir, 'README.md')
    print(f"Creating README at: {readme_path}")
    
    with open(readme_path, 'w') as f:
        f.write("# MNIST Data Augmentation Examples\n\n")
        f.write("This directory contains examples of data augmentation applied to MNIST digits.\n\n")
        f.write("## Directory Structure\n")
        f.write("```\n")
        f.write("augmentations/\n")
        f.write("├── digit_0/         # Augmentations for digit 0\n")
        f.write("├── digit_1/         # Augmentations for digit 1\n")
        f.write("└── digit_2/         # Augmentations for digit 2\n")
        f.write("```\n\n")
    
    # Get sample images
    print("Getting sample images...")
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Get samples of different digits
    samples = []
    labels = []
    seen_digits = set()
    
    for img, label in dataset:
        if label not in seen_digits:
            samples.append(img)
            labels.append(label)
            seen_digits.add(label)
            print(f"Found digit: {label}")
            if len(seen_digits) == 3:
                break
    
    # Generate visualizations for each sample
    for idx, (img, label) in enumerate(zip(samples, labels)):
        print(f"\nGenerating augmentations for digit {label}...")
        visualize_augmentations(img, num_samples=5, digit=label)
        
        # Add link to main README
        with open(readme_path, 'a') as f:
            f.write(f"\n## Digit {label}\n")
            f.write(f"See [detailed augmentations for digit {label}](digit_{label}/README.md)\n")
    
    print("\nAugmentation generation completed.")

def test_lr_scheduler():
    """Test OneCycleLR scheduler behavior"""
    model = MNISTNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Create a small dummy dataloader for testing
    batch_size = 4
    n_batches = 10
    total_steps = n_batches
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=1,
        steps_per_epoch=n_batches,
        pct_start=0.25,
        div_factor=10,
        final_div_factor=100
    )
    
    # Track learning rates
    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    # Verify learning rate behavior
    assert lrs[0] < lrs[n_batches//4], "LR should increase during warmup"
    assert lrs[n_batches//4] > lrs[-1], "LR should decrease after warmup"
    assert max(lrs) <= 0.1, "Max LR should not exceed specified max_lr"
    print(f"\nLR Schedule Test: Peak LR = {max(lrs):.4f}")

def test_augmentation_consistency():
    """Test that augmentations maintain digit recognizability"""
    # Force CPU
    device = torch.device('cpu')
    
    # Define transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    augment = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ])
    
    # Load model and sample image
    model = MNISTNet().to(device)
    model_path = os.environ.get('TRAINED_MODEL_PATH')
    if not model_path:
        model_files = glob.glob('models/*.pth')
        if not model_files:
            model_path = train()
        else:
            model_path = max(model_files, key=os.path.getctime)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get a sample image
    dataset = datasets.MNIST('./data', train=False, download=True, transform=to_tensor)
    sample_tensor, original_label = dataset[0]
    
    # Convert to PIL for augmentations
    sample_pil = to_pil(sample_tensor)
    
    # Test multiple augmentations
    n_augmentations = 10
    correct_predictions = 0
    
    with torch.no_grad():
        for _ in range(n_augmentations):
            # Apply augmentations to PIL Image
            augmented_pil = augment(sample_pil)
            # Convert back to tensor for model
            augmented_tensor = to_tensor(augmented_pil).unsqueeze(0)
            
            output = model(augmented_tensor)
            pred = output.argmax(dim=1).item()
            correct_predictions += (pred == original_label)
    
    accuracy = correct_predictions / n_augmentations
    print(f"\nAugmentation Consistency: {accuracy*100:.1f}% correct predictions")
    assert accuracy >= 0.7, "Model should maintain >70% accuracy on augmented images"

def test_noise_robustness():
    """Test model's robustness to input noise"""
    # Force CPU
    device = torch.device('cpu')
    
    model = MNISTNet().to(device)
    model_path = os.environ.get('TRAINED_MODEL_PATH')
    if not model_path:
        model_files = glob.glob('models/*.pth')
        if not model_files:
            model_path = train()
        else:
            model_path = max(model_files, key=os.path.getctime)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get test sample
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    image, label = dataset[0]
    
    # Test different noise levels
    noise_levels = [0.1, 0.2, 0.3]
    results = []
    
    with torch.no_grad():
        # Get baseline prediction
        baseline_output = model(image.unsqueeze(0))
        baseline_pred = baseline_output.argmax(dim=1).item()
        
        # Test with different noise levels
        for noise_level in noise_levels:
            noise = torch.randn_like(image) * noise_level
            noisy_image = torch.clamp(image + noise, 0, 1)
            output = model(noisy_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            results.append(pred == baseline_pred)
    
    # Model should be robust to at least the lowest noise level
    assert results[0], f"Model should be robust to {noise_levels[0]} noise level"
    print(f"\nNoise Robustness: Maintained prediction up to {max([l for l, r in zip(noise_levels, results) if r])} noise level")

if __name__ == "__main__":
    test_model()