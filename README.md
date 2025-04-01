# Medical Image Classification Project

A PyTorch-based medical image classification project using the MedMNIST dataset. The project supports multiple deep learning model architectures, including simple CNN and ResNet series models.

## Project Structure

```
.
├── data/                   # Dataset directory
├── models/                 # Model definitions
│   ├── simple_cnn.py      # Simple CNN model
│   └── resnet.py          # ResNet models
├── utils/                  # Utility functions
│   └── data_utils.py      # Data processing utilities
├── run.py                 # Training script
├── test.py                # Testing script
├── predict.py             # Prediction script
├── requirements.txt       # Project dependencies
└── LICENSE               # MIT License
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- medmnist
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone [repository_url]
cd [project_directory]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Models

Use `run.py` to train models:

```bash
python run.py
```

You can configure training by modifying the following parameters in `run.py`:
- `data_flag`: Select dataset ('pathmnist', 'chestmnist', 'dermamnist', 'octmnist', etc.)
- `model_name`: Choose model ('cnn', 'resnet18', 'resnet34')
- `batch_size`: Batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate

### 2. Testing Models

Use `test.py` to evaluate model performance:

```bash
python test.py
```

The testing script will:
- Load the trained model
- Evaluate model performance on the test set
- Generate confusion matrix visualization
- Save test metrics to file

### 3. Predicting New Images

Use `predict.py` to predict single images:

```bash
python predict.py
```

The prediction script will:
- Randomly select a sample image from the test set
- Make predictions using the trained model
- Generate prediction visualization
- Save prediction details to file

## Supported Models

1. Simple CNN Model (`cnn`)
   - Suitable for basic image classification tasks
   - Contains convolutional, pooling, and fully connected layers

2. ResNet18 Model (`resnet18`)
   - Deep convolutional neural network with residual connections
   - 18-layer network structure
   - Suitable for medium-complexity classification tasks

3. ResNet34 Model (`resnet34`)
   - Deeper ResNet variant
   - 34-layer network structure
   - Suitable for complex classification tasks

## Dataset Information

The MedMNIST dataset includes multiple medical image sub-datasets:
- PathMNIST: Pathological images
- ChestMNIST: Chest X-ray images
- DermaMNIST: Skin lesion images
- OCTMNIST: Optical coherence tomography images

Each dataset has its specific:
- Number of channels (grayscale/RGB)
- Number of classes
- Task type (single-label/multi-label classification)

## Output Files

1. Training Process:
   - `best_model_{model_name}.pth`: Saved best model weights

2. Testing Results:
   - `test_metrics_{model_name}.txt`: Test metrics (accuracy, precision, recall, F1 score)
   - `confusion_matrix_{model_name}.png`: Confusion matrix visualization

3. Prediction Results:
   - `sample_image.png`: Sample image used for prediction
   - `prediction_{model_name}.png`: Prediction result visualization
   - `prediction_details_{model_name}.txt`: Detailed prediction information

## Notes

1. Ensure the model is trained before running the prediction script
2. Choose appropriate model architecture and dataset based on your needs
3. Adjust batch size and training epochs for large datasets
4. Ensure sufficient GPU memory for running ResNet models

## Contributing

Issues and improvement suggestions are welcome!

## License

[MIT](LICENSE)