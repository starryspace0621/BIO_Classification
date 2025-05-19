# Medical Image Classification Project

A PyTorch-based medical image classification project using the MedMNIST dataset. The project supports multiple deep learning model architectures, including simple CNN and ResNet series models, and supports both 2D and 3D medical image datasets.

## Project Structure

```
.
├── data/                   # Dataset directory
├── models/                 # Model definitions
│   ├── simple_cnn.py      # Simple CNN model
│   └── resnet.py          # ResNet models
├── layers/                # Custom layer definitions
├── saved_models/          # Saved model weights
├── results/               # Prediction and test results
├── utils/                 # Utility functions
│   └── data_utils.py      # Data processing utilities
├── web_app/              # Web application for model deployment
├── run.py                # Training script
├── test.py               # Testing script
├── predict.py            # Prediction script
├── requirements.txt      # Project dependencies
└── LICENSE              # MIT License
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- medmnist
- scikit-learn
- matplotlib
- seaborn
- Flask (for web application)

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
python run.py --data_flag <dataset_name> --model_name <model_name> --batch_size <batch_size> --num_epochs <num_epochs> --learning_rate <learning_rate>
```

You can configure training by modifying the following parameters:
- `data_flag`: Select dataset (see Dataset Information section below)
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
python predict.py --data_flag <dataset_name> --model_name <model_name> --image_path <image_path>
```

The prediction script will:
- Load and preprocess the input image
- Make predictions using the trained model
- Generate prediction visualization
- Save prediction details to file

### 4. Web Application

The project includes a web application for easy model deployment and prediction. The web application provides an intuitive user interface for both training and prediction tasks.

#### Directory Structure
```
web_app/
├── app.py              # Flask application
├── static/            # Static files
│   ├── css/          # CSS styles
│   ├── js/           # JavaScript files
│   ├── images/       # Images and icons
│   └── uploads/      # Uploaded images for prediction
└── templates/        # HTML templates
    └── index.html    # Main page template
```

#### Features

1. Training Mode:
   - Dataset selection (2D and 3D)
   - Model selection (CNN, ResNet18, ResNet34)
   - Training parameter configuration
     - Batch size
     - Number of epochs
     - Learning rate
   - Real-time training progress monitoring
     - Current epoch
     - Training accuracy
     - Learning rate
   - Training completion notification

2. Prediction Mode:
   - Dataset selection
   - Pre-trained model selection
   - Image upload support
     - 2D images (PNG, JPG)
     - 3D images (NPY)
   - Real-time prediction results
     - Top 3 predictions with probabilities
     - Visual representation of results
   - Result confirmation

3. User Interface:
   - Responsive design using Bootstrap
   - Interactive forms with validation
   - Real-time progress updates using WebSocket
   - Toast notifications for status updates
   - Support for both desktop and mobile devices

#### Running the Web Application

1. Quick Start (Windows):
   - Double-click `web_app/start_web.bat`
   - The script will:
     - Check Python installation
     - Install required packages
     - Create necessary directories
     - Start the web server
   - Open your browser and go to `http://localhost:5000`

2. Manual Start:
   - Install additional requirements:
   ```bash
   pip install flask flask-socketio
   ```

   - Start the web server:
   ```bash
   cd web_app
   python app.py
   ```

   - Access the application:
     - Open your web browser
     - Navigate to `http://localhost:5000`

#### Notes
- The web application requires a trained model for prediction
- For 3D datasets, ensure the uploaded files are in the correct format
- The application automatically handles both 2D and 3D models
- Training progress is saved and can be monitored in real-time
- Uploaded images are temporarily stored in the `static/uploads` directory

## Supported Models

1. Simple CNN Model (`cnn`)
   - Suitable for basic image classification tasks
   - Contains convolutional, pooling, and fully connected layers
   - Available in both 2D and 3D versions

2. ResNet18 Model (`resnet18`)
   - Deep convolutional neural network with residual connections
   - 18-layer network structure
   - Available in both 2D and 3D versions
   - Suitable for medium-complexity classification tasks

3. ResNet34 Model (`resnet34`)
   - Deeper ResNet variant
   - 34-layer network structure
   - Available in both 2D and 3D versions
   - Suitable for complex classification tasks

## Dataset Information

The project supports both 2D and 3D medical image datasets from MedMNIST:

### 2D Datasets
- PathMNIST: Pathological images
- ChestMNIST: Chest X-ray images
- DermaMNIST: Skin lesion images
- OCTMNIST: Optical coherence tomography images
- PneumoniaMNIST: Pneumonia X-ray images
- RetinaMNIST: Retina fundus images
- BreastMNIST: Breast ultrasound images
- BloodMNIST: Blood cell images
- TissueMNIST: Tissue images
- OrganAMNIST: Organ axial images
- OrganCMNIST: Organ coronal images
- OrganSMNIST: Organ sagittal images

### 3D Datasets
- OrganMNIST3D: 3D organ images
- NoduleMNIST3D: 3D lung nodule images
- AdrenalMNIST3D: 3D adrenal gland images
- FractureMNIST3D: 3D bone fracture images
- VesselMNIST3D: 3D blood vessel images
- SynapseMNIST3D: 3D synapse images

Each dataset has its specific:
- Number of channels (grayscale/RGB)
- Number of classes
- Task type (single-label/multi-label classification)
- Dimensionality (2D/3D)

## Output Files

1. Training Process:
   - `saved_models/<data_flag>/best_model_<data_flag>_<model_name>.pth`: Saved best model weights

2. Testing Results:
   - `test_metrics_<model_name>.txt`: Test metrics (accuracy, precision, recall, F1 score)
   - `confusion_matrix_<model_name>.png`: Confusion matrix visualization

3. Prediction Results:
   - `results/<data_flag>/<model_name>/prediction_<data_flag>_<model_name>.png`: Prediction result visualization
   - `results/<data_flag>/<model_name>/prediction_details_<data_flag>_<model_name>.txt`: Detailed prediction information

## Notes

1. Ensure the model is trained before running the prediction script
2. Choose appropriate model architecture and dataset based on your needs
3. For 3D datasets, ensure sufficient GPU memory and computational resources
4. The web application requires Flask to be installed
5. Adjust batch size and training epochs based on dataset size and available resources

## Contributing

Issues and improvement suggestions are welcome!

## License

[MIT](LICENSE)