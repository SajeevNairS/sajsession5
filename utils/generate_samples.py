import os
import sys
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_sample_sheet():
    """Generate a sample sheet of augmented images"""
    # Create output directory
    os.makedirs('visualizations/augmented_samples', exist_ok=True)
    
    # Get sample images
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Get one sample of digit 5
    for img, label in dataset:
        if label == 5:  # Let's use digit 5 as example
            sample_image = img
            break
    
    # Define augmentations
    augmentations = [
        ("Original", None),
        ("Rotation", transforms.RandomRotation(degrees=5)),
        ("Affine", transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))),
        ("Combined", transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ]))
    ]
    
    # Create sample sheet
    fig, axes = plt.subplots(len(augmentations), 5, figsize=(15, 12))
    fig.suptitle('MNIST Digit Augmentation Samples', size=16)
    
    for row, (aug_name, transform) in enumerate(augmentations):
        axes[row, 0].set_ylabel(aug_name, size=10, rotation=0, ha='right')
        
        for col in range(5):
            ax = axes[row, col]
            
            if transform is None:
                img_show = sample_image
            else:
                img_show = transform(sample_image)
            
            ax.imshow(img_show.squeeze(), cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save sample sheet
    sample_sheet_path = 'visualizations/augmented_samples/augmentation_samples.png'
    plt.savefig(sample_sheet_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Create README with the sample sheet
    with open('visualizations/augmented_samples/README.md', 'w') as f:
        f.write("# MNIST Augmentation Samples\n\n")
        f.write("This shows different augmentations applied to a single MNIST digit:\n\n")
        f.write("- Row 1: Original image\n")
        f.write("- Row 2: Random rotation (±5°)\n")
        f.write("- Row 3: Random affine (rotation, translation, scale)\n")
        f.write("- Row 4: Combined augmentations\n\n")
        f.write("![Augmentation Samples](augmentation_samples.png)\n")

if __name__ == "__main__":
    generate_sample_sheet()
    print("Generated augmentation samples in visualizations/augmented_samples/") 