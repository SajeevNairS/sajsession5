import sys
import os
import torch
import pytest
from torchvision import datasets, transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mnist_model import MNISTNet
from train import train

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
    summary.append("## Per-Class Performance")
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
    # Parameter count test
    model = MNISTNet()
    param_count = count_parameters(model)
    
    # Accuracy test
    save_path = train()
    model.load_state_dict(torch.load(save_path))
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

if __name__ == "__main__":
    test_model()