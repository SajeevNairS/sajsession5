import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os

def visualize_augmentations(image, num_samples=5, digit=None):
    """Visualize different augmentations of an image"""
    try:
        # Create specific folders for each digit
        digit_str = f"digit_{digit}"
        digit_folder = f'visualizations/augmentations/{digit_str}'
        os.makedirs(digit_folder, exist_ok=True)
        
        # Define augmentations
        augmentations = [
            ("Rotation", transforms.RandomRotation(degrees=5)),
            ("Affine", transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))),
            ("Combined", transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))
            ]))
        ]
        
        # Save original image with relative path for GitHub
        plt.figure(figsize=(4, 4))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Original Digit {digit}')
        plt.axis('off')
        orig_path = f'{digit_str}/original.png'
        plt.savefig(f'visualizations/augmentations/{orig_path}', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Create a summary markdown for this digit
        with open(f'{digit_folder}/README.md', 'w') as f:
            f.write(f"# Augmentations for Digit {digit}\n\n")
            f.write("## Original Image\n")
            f.write(f"![Original]({orig_path})\n\n")
        
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
                
                # Save individual samples with relative paths
                sample_path = f'{digit_str}/{aug_name.lower()}_sample_{i+1}.png'
                plt.imsave(
                    f'visualizations/augmentations/{sample_path}',
                    augmented.squeeze().numpy(),
                    cmap='gray'
                )
            
            # Save grid with relative path
            grid_path = f'{digit_str}/{aug_name.lower()}_grid.png'
            plt.tight_layout()
            plt.savefig(f'visualizations/augmentations/{grid_path}',
                       bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            # Add to digit's README with relative paths
            with open(f'{digit_folder}/README.md', 'a') as f:
                f.write(f"\n## {aug_name} Augmentation\n")
                f.write(f"![{aug_name} Grid]({aug_name.lower()}_grid.png)\n\n")
                f.write("### Individual Samples\n")
                for i in range(num_samples):
                    f.write(f"![Sample {i+1}]({aug_name.lower()}_sample_{i+1}.png) ")
                f.write("\n")
        
        return True
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return False
    