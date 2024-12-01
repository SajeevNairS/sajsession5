# MNIST Digit Classification with Lightweight CNN

A PyTorch implementation of a lightweight CNN for MNIST digit classification, optimized to achieve >95% accuracy in one epoch with under 25,000 parameters.

## Model Architecture

### Network Design
- Progressive channel expansion: 1 → 4 → 8 → 12 → 16 → 20 → 24 → 28
- Multi-stage spatial reduction through pooling layers
- GELU activation for better gradient flow
- BatchNorm for training stability
- Dropout for regularization

### Layer Structure
1. **Input Layer**: 1×28×28 grayscale images
2. **Convolutional Blocks**:
   - Conv1: Basic edge detection (3×3 kernel, 1→4 channels)
   - Conv2: Simple pattern detection (3×3 kernel, 4→8 channels)
   - Conv3: Pattern combinations (3×3 kernel, 8→12 channels)
   - Conv4: Initial feature aggregation (3×3 kernel, 12→16 channels)
   - Conv5: Mid-level features (3×3 kernel, 16→20 channels)
   - Conv6: High-level features (3×3 kernel, 20→24 channels)
   - Conv7: Global features (3×3 kernel, 24→28 channels)
3. **Spatial Reduction**: 28×28 → 14×14 → 7×7 → 4×4 → 2×2
4. **Classification Head**: 28 → 56 → 10

### Receptive Field
- Progressive growth through layers
- Final receptive field: 31×31 (covers entire input)
- Layer-wise RF growth documented in model architecture

## Training Setup

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Training Configuration
- Optimizer: SGD with momentum
  * Learning rate: 0.001
  * Momentum: 0.9
- Batch size: 4
- One Cycle LR Schedule:
  * Max LR: 0.1
  * Warmup: 25% of training
  * Cosine annealing
- Gradient clipping: 1.0

## Testing and Verification

### Automated Tests
The project includes automated tests to verify:
1. Parameter count (< 25,000)
2. Model accuracy (≥ 95%)
3. Model architecture
4. Per-class performance

### Running Tests Locally
```bash
# Run all tests
python tests/test_model.py

# Run with pytest
pytest tests/test_model.py -v -s

# View test summary
cat test-summary.md
```

### Test Artifacts
- test-summary.md: Detailed test results
- visualizations/: Feature maps and model architecture
- models/: Saved model checkpoints

### GitHub Actions
Automated CI/CD pipeline that:
- Verifies parameter count
- Checks model accuracy
- Generates test summary
- Posts results as PR comments
- Saves test artifacts

## Project Structure
```
.
├── model/
│   └── mnist_model.py     # Model architecture
├── tests/
│   └── test_model.py      # Test suite
├── visualizations/        # Generated visualizations
├── train.py              # Training script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Requirements
- torch
- torchvision
- numpy
- matplotlib
- pytest
- Python 3.6+

## Usage

### Training
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py
```

### Testing
```bash
# Run tests
python tests/test_model.py

# View results
cat test-summary.md
```

## Results
- Parameters: 18,694 (under 25K limit)
- Accuracy: >95% in one epoch
- Training time: Single CPU
- Full test results in test-summary.md

## Data Augmentation

Augmented samples are available in:
- `visualizations/augmentations/` - Contains augmented images and documentation
- Individual digit folders with original and augmented samples
- See [augmentation examples](visualizations/augmentations/README.md)



