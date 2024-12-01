import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os

def visualize_augmentations(image, num_samples=5):
    """
    Visualize different augmentations of an image
    Args:
        image: Original image tensor
        num_samples: Number of augmented samples to generate
    """
    print("Starting augmentation visualization...")
    
    # Define augmentations used in training
    augmentations = [
        ("Rotation", transforms.RandomRotation(degrees=5)),
        ("Resize", transforms.Resize((28, 28))),
        ("Training", transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ]))
    ]
    
    print("Saving original image...")
    # Save original image
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig('visualizations/augmentations/original.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print("Generating augmented samples...")
    # Generate and save augmented samples
    for aug_name, aug_transform in augmentations:
        print(f"Processing {aug_name} augmentation...")
        plt.figure(figsize=(15, 3))
        
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            augmented = aug_transform(image)
            plt.imshow(augmented.squeeze(), cmap='gray')
            plt.title(f'{aug_name} {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        save_path = f'visualizations/augmentations/{aug_name.lower()}_samples.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved {save_path}")
        plt.close()
    
    print("Creating summary markdown...")
    # Create summary markdown
    with open('visualizations/augmentations/README.md', 'w') as f:
        f.write("# Data Augmentation Samples\n\n")
        f.write("## Original Image\n")
        f.write("![Original](original.png)\n\n")
        
        for aug_name, _ in augmentations:
            f.write(f"## {aug_name} Augmentation\n")
            f.write(f"![{aug_name}]({aug_name.lower()}_samples.png)\n\n")
    
    print("Augmentation visualization completed successfully.")
    