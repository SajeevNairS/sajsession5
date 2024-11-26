# MNIST Classification with CI/CD Pipeline

This project demonstrates a complete CI/CD pipeline for a machine learning project using PyTorch and GitHub Actions. It includes a simple convolutional neural network (CNN) for MNIST digit classification with automated testing, model validation, and deployment processes.


## Model Architecture
The model is a simple CNN with the following architecture:
- Input Layer: Accepts 28x28 grayscale images
- Conv Layer 1: 16 filters, 3x3 kernel, ReLU activation + MaxPool
- Conv Layer 2: 32 filters, 3x3 kernel, ReLU activation + MaxPool
- Fully Connected Layer 1: 1568 -> 128 units, ReLU activation
- Output Layer: 128 -> 10 units (one for each digit)

## Features
- Automated model training pipeline
- Model validation checks:
  - Input shape verification (28x28)
  - Parameter count validation (< 100,000)
  - Output dimension check (10 classes)
  - Model accuracy verification (> 80%)
- Automated model versioning with timestamps
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


5. Run tests:

bash
pytest tests/


### GitHub Actions Pipeline
The CI/CD pipeline automatically runs on every push to the repository. To use it:

1. Fork this repository
2. Enable GitHub Actions in your repository settings
3. Push your changes to trigger the pipeline

The pipeline will:
1. Set up a Python environment
2. Install dependencies
3. Train the model
4. Run validation tests
5. Save the trained model as an artifact

## Model Artifacts
Trained models are saved with timestamps in the format:

mnist_model_YYYYMMDD_HHMMSS.pth

You can find these:
- Locally in the `models/` directory after training
- In GitHub Actions artifacts after pipeline completion

## Testing
The project includes automated tests that verify:
- Model architecture compliance
- Input/output shape compatibility
- Parameter count constraints
- Model performance (accuracy > 80%)

Run tests locally using:

bash
pytest tests/


## Contributing
1. Fork the repository
2. Create your feature branch:

bash
git checkout -b feature/amazing-feature

3. Commit your changes:

bash
git commit -m 'Add some amazing feature'

4. Push to the branch:

bash
git push origin feature/amazing-feature

5. Open a Pull Request

## Troubleshooting
- If you encounter CUDA/GPU issues, ensure the model runs on CPU by default
- For memory issues, reduce batch size in `train.py`
- For test failures, check model accuracy and architecture constraints

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- MNIST Dataset: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- GitHub Actions for CI/CD automation

## Contact
For questions or feedback, please open an issue in the GitHub repository.

