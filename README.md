# MNIST Digit Classification with Lightweight CNN

[![ML Pipeline](https://github.com/SajeevNairS/sajsession5/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/SajeevNairS/sajsession5/actions/workflows/ml_pipeline.yml)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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

## Training Setup

### Data Augmentation
The model uses robust data augmentation techniques:
- Random rotation (±5°)
- Random affine transformations:
  * Translation: ±5% of image size
  * Scale: 95-105% of original size
- Combined transformations for increased robustness

```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
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

## Model Testing

### Automated Tests
The project includes comprehensive tests:
1. **Parameter Count**: Verifies model size < 25,000 parameters
2. **Accuracy Test**: Ensures ≥95% accuracy on test set
3. **Learning Rate Schedule Test**: Validates proper LR behavior
4. **Augmentation Consistency Test**: Checks robustness to transformations
5. **Noise Robustness Test**: Verifies stability under input noise

### Test Results
- Parameters: 18,694 (well under 25K limit)
- Base Accuracy: 98.90%
- Augmentation Consistency: 90.0%
- Noise Tolerance: Up to 0.2 noise level
- Per-class Accuracy: >97.8% for all digits

### Running Tests
```bash
# Run all tests
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py -k "test_augmentation_consistency" -v
```

## Project Structure
```
.
├── model/
│   └── mnist_model.py     # Model architecture
├── tests/
│   └── test_model.py      # Comprehensive test suite
├── utils/
│   ├── augmentation_viz.py  # Augmentation visualization
│   └── generate_samples.py  # Sample generation
├── visualizations/        # Generated visualizations
├── train.py              # Training script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Requirements
- torch (CPU version)
- torchvision
- numpy
- matplotlib
- pytest
- Python 3.8+

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
# Run all tests
python tests/test_model.py

# Generate augmentation samples
python utils/generate_samples.py

# View results
cat test-summary.md
```

## CI/CD Pipeline
The project includes a GitHub Actions workflow that:
1. Trains the model on CPU
2. Runs comprehensive test suite
3. Verifies model requirements
4. Generates test summary
5. Saves artifacts and visualizations

## Results Summary
- Parameters: 18,694 (25.2% below limit)
- Accuracy: 98.90% (+3.90% above target)
- Training: Single epoch on CPU
- Augmentation: 90.0% consistency
- Robustness: Tolerates up to 0.2 noise level

For detailed test results and visualizations, see [test-summary.md](test-summary.md).



