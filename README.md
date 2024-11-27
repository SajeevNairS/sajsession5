# MNIST Classification with CI/CD Pipeline

This project demonstrates a complete CI/CD pipeline for a machine learning project using PyTorch and GitHub Actions. It includes a lightweight convolutional neural network (CNN) for MNIST digit classification with automated testing, model validation, and deployment processes.

## Project Structure

.
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # CI/CD pipeline configuration
├── model/
│ ├── init.py
│ └── mnist_model.py # Neural network architecture
├── tests/
│ └── test_model.py # Model validation tests
├── train.py # Training script
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules


## Model Architecture
The model is an efficient CNN with the following architecture:
- Input Layer: 28x28 grayscale images
- Conv Layer 1: 4 filters, 3x3 kernel, BatchNorm, ReLU, MaxPool
- Conv Layer 2: 8 filters, 3x3 kernel, BatchNorm, ReLU, MaxPool
- Fully Connected Layer 1: 392 -> 16 units, ReLU
- Output Layer: 16 -> 10 units (one for each digit)
- Dropout (0.3) after each layer for regularization

Total Parameters: 6,818 (well under the 25,000 parameter limit)

## Features
- Single epoch training achieving >95% accuracy
- Model validation checks:
  - Input shape verification (28x28)
  - Parameter count validation
  - Output dimension check (10 classes)
  - Model accuracy verification
- Automated model versioning with timestamps and accuracy
- CPU-only PyTorch for wider compatibility
- GitHub Actions integration for CI/CD

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Local Development
1. Clone the repository:

bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


2. Create and activate a virtual environment:

bash
On Unix/macOS
python -m venv venv
source venv/bin/activate
On Windows
python -m venv venv
venv\Scripts\activate


3. Install dependencies:

bash
pip install -r requirements.txt


4. Train the model:

bash
python train.py

You'll see progress updates showing:
- Current batch number
- Loss value
- Running accuracy
- Final training accuracy

5. Run tests:

bash
pytest tests/


### GitHub Actions Pipeline
The CI/CD pipeline automatically runs on every push to the repository and performs:
1. Environment setup with Python 3.8
2. Installation of CPU-only PyTorch dependencies
3. Single epoch model training
4. Validation tests
5. Model artifact storage

## Model Artifacts
Trained models are saved with timestamps and accuracy in the format:

mnist_model_YYYYMMDD_HHMMSS_acc95.0.pth

These can be found in:
- `models/` directory (local training)
- GitHub Actions artifacts (pipeline runs)

## Testing
The automated tests verify:
- Model architecture compliance
- Input/output shape compatibility
- Parameter count (< 100,000)
- Model performance (> 80% accuracy)

## Troubleshooting
- If you encounter import errors, ensure you're in the project root directory
- For memory issues, reduce batch size in `train.py` (currently 64)
- All PyTorch operations run on CPU by default

## License
MIT License

## Acknowledgments
- MNIST Dataset: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- GitHub Actions for CI/CD automation

## Contact
For questions or feedback, please open an issue in the GitHub repository.

