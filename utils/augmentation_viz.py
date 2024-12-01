import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os

def visualize_augmentations(image, num_samples=5, digit=None):
    """
    Visualize different augmentations of an image
    Args:
        image: Original image tensor
        num_samples: Number of augmented samples to generate
        digit: The digit label (for filename)
    """
    digit_str = f"digit_{digit}" if digit is not None else "sample"
    
    # Define augmentations used in training
    augmentations = [
        ("Rotation", transforms.RandomRotation(degrees=5)),
        ("Affine", transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))),
        ("Combined", transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ]))
    ]
    
    # Save original image
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Original Digit {digit}')
    plt.axis('off')
    plt.savefig(f'visualizations/augmentations/original_{digit_str}.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Generate and save augmented samples
    for aug_name, aug_transform in augmentations:
        plt.figure(figsize=(15, 3))
        augmented_samples = []
        
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            augmented = aug_transform(image)
            augmented_samples.append(augmented)
            plt.imshow(augmented.squeeze(), cmap='gray')
            plt.title(f'{aug_name} {i+1}')
            plt.axis('off')
            
            # Save individual augmented samples
            plt.imsave(
                f'visualizations/augmented_samples/{digit_str}_{aug_name.lower()}_{i+1}.png',
                augmented.squeeze().numpy(),
                cmap='gray'
            )
        
        plt.tight_layout()
        plt.savefig(f'visualizations/augmentations/{aug_name.lower()}_{digit_str}_samples.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    # Create summary markdown for this digit
    with open(f'visualizations/augmentations/README_{digit_str}.md', 'w') as f:
        f.write(f"# Data Augmentation Samples for Digit {digit}\n\n")
        f.write("## Original Image\n")
        f.write(f"![Original](original_{digit_str}.png)\n\n")
        
        for aug_name, _ in augmentations:
            f.write(f"## {aug_name} Augmentation\n")
            f.write(f"![{aug_name}]({aug_name.lower()}_{digit_str}_samples.png)\n\n")
    
    # Create main README if it doesn't exist
    if not os.path.exists('visualizations/augmentations/README.md'):
        with open('visualizations/augmentations/README.md', 'w') as f:
            f.write("# MNIST Data Augmentation Examples\n\n")
            f.write("This directory contains examples of data augmentation applied to MNIST digits.\n\n")
            f.write("## Directory Structure\n")
            f.write("- `augmented_samples/`: Individual augmented images\n")
            f.write("- `README_digit_X.md`: Detailed augmentations for each digit\n")
            f.write("- Various .png files showing augmentation results\n")
    