import torch
from torchvision import datasets, transforms
from augmentation_viz import visualize_augmentations
import os

def main():
    print("Starting augmentation visualization generation...")
    
    # Create directories
    os.makedirs('visualizations/augmentations', exist_ok=True)
    
    # Get sample image
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sample_image = dataset[0][0]
    
    print("Got sample image, generating augmentations...")
    
    # Generate visualizations
    visualize_augmentations(sample_image, num_samples=5)
    
    print("Augmentation visualization completed.")

if __name__ == "__main__":
    main() 