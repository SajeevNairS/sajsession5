import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    Lightweight CNN architecture for MNIST digit classification.
    
    Architecture highlights:
    - Progressive channel expansion (1->4->8->12->16->20->24->28)
    - Multi-stage spatial reduction through pooling layers
    - GELU activation for better gradient flow
    - BatchNorm for training stability
    - Under 25K parameters
    
    Receptive field calculation:
    - Each conv layer (3x3 kernel): adds 2 to RF
    - Each pooling (2x2, stride 2): doubles RF
    - Final RF: 31x31 (covers entire input image)
        * Conv1: 3x3
        * Conv2: 5x5
        * Conv3: 7x7
        * Conv4 + Pool4: 11x11
        * Conv5 + Pool5: 15x15
        * Conv6 + Pool6: 23x23
        * Conv7 + Pool7: 31x31
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First block: Basic edge detection
        # RF: 3x3
        # Input: [batch, 1, 28, 28] -> Output: [batch, 4, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)  # Stabilize learning
        
        # Second block: Simple patterns
        # RF: 5x5
        # Input: [batch, 4, 28, 28] -> Output: [batch, 8, 28, 28]
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # Third block: Pattern combinations
        # RF: 7x7
        # Input: [batch, 8, 28, 28] -> Output: [batch, 12, 28, 28]
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(12)
        
        # Fourth block: Initial feature aggregation
        # RF: 11x11 after pooling
        # Input: [batch, 12, 28, 28] -> Output: [batch, 16, 14, 14]
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # First spatial reduction
        
        # Fifth block: Mid-level features
        # RF: 15x15 after pooling
        # Input: [batch, 16, 14, 14] -> Output: [batch, 20, 7, 7]
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(20)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Second spatial reduction
        
        # Sixth block: High-level features
        # RF: 23x23 after pooling
        # Input: [batch, 20, 7, 7] -> Output: [batch, 24, 4, 4]
        self.conv6 = nn.Conv2d(in_channels=20, out_channels=24, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(24)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)  # Third spatial reduction
        
        # Seventh block: Global features
        # RF: 31x31 after pooling
        # Input: [batch, 24, 4, 4] -> Output: [batch, 28, 2, 2]
        self.conv7 = nn.Conv2d(in_channels=24, out_channels=28, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(28)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)  # Final spatial reduction
        
        # Global feature processing
        self.flatten = nn.Flatten()  # [batch, 28, 2, 2] -> [batch, 28]
        
        # Classification head
        self.fc1 = nn.Linear(28, 56)  # Feature space expansion
        self.fc2 = nn.Linear(56, 10)  # Digit classification (10 classes)
        self.dropout = nn.Dropout(0.15)  # Light regularization

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 1, 28, 28]
                            Values normalized to mean=0.1307, std=0.3081
        
        Returns:
            torch.Tensor: Logits for each digit class [batch_size, 10]
        
        Feature extraction process:
        1. Basic edge detection (3x3 RF)
        2. Simple pattern detection (5x5 RF)
        3. Pattern combinations (7x7 RF)
        4. Initial feature aggregation + pooling (11x11 RF)
        5. Mid-level features + pooling (15x15 RF)
        6. High-level features + pooling (23x23 RF)
        7. Global features + pooling (31x31 RF)
        """
        # Progressive feature extraction
        x = F.gelu(self.bn1(self.conv1(x)))  # Basic edges
        x = F.gelu(self.bn2(self.conv2(x)))  # Simple patterns
        x = F.gelu(self.bn3(self.conv3(x)))  # Pattern combinations
        
        x = F.gelu(self.bn4(self.conv4(x)))  # Initial features
        x = self.pool4(x)                     # 28x28 -> 14x14
        
        x = F.gelu(self.bn5(self.conv5(x)))  # Mid-level features
        x = self.pool5(x)                     # 14x14 -> 7x7
        
        x = F.gelu(self.bn6(self.conv6(x)))  # High-level features
        x = self.pool6(x)                     # 7x7 -> 4x4
        
        x = F.gelu(self.bn7(self.conv7(x)))  # Global features
        x = self.pool7(x)                     # 4x4 -> 2x2
        
        # Classification
        x = x.view(-1, 28)                    # Flatten features
        x = F.gelu(self.fc1(x))              # Feature transformation
        x = self.fc2(x)                       # Class prediction
        return x

    def __repr__(self):
        """
        String representation showing model architecture and parameters.
        """
        return (
            f"MNISTNet(\n"
            f"  Input: 1x28x28\n"
            f"  Conv blocks: 1->4->8->12->16->20->24->28 channels\n"
            f"  Spatial reduction: 28->14->7->4->2\n"
            f"  Receptive field: 31x31\n"
            f"  Classification: 28->56->10\n"
            f"  Activation: GELU\n"
            f"  Parameters: {sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )