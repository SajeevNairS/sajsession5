import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os

def visualize_augmentations(image, num_samples=5, digit=None):
    """Visualize different augmentations of an image"""
    try:
        digit_str = f"digit_{digit}" if digit is not None else "sample"
        
        # Define augmentations
        augmentations = [
            ("Rotation", transforms.RandomRotation(degrees=5)),
            ("Affine", transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))),
            ("Combined", transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))
            ]))
        ]
        
        # Save original image
        plt.figure(figsize=(4, 4))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Original Digit {digit}')
        plt.axis('off')
        plt.savefig(f'visualizations/augmentations/original_{digit_str}.png', 
                   bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Generate augmented samples
        for aug_name, aug_transform in augmentations:
            fig = plt.figure(figsize=(15, 3))
            fig.suptitle(f'{aug_name} Augmentations', y=1.05)
            for i in range(num_samples):
                plt.subplot(1, num_samples, i+1)
                augmented = aug_transform(image)
                plt.imshow(augmented.squeeze(), cmap='gray')
                plt.title(f'Sample {i+1}')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'visualizations/augmentations/{aug_name.lower()}_{digit_str}.png',
                       bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
        return True
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return False
    