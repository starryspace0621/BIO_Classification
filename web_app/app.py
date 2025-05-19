import os
import sys
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from medmnist import INFO
import json
from run import get_model, train_model
from predict import predict_image
import numpy as np
from PIL import Image
import io
import medmnist
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Get all available datasets
DATASETS = ['pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 
            'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 
            'organcmnist', 'organsmnist', 'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
            'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']

# Classify 2D and 3D datasets
DATASETS_2D = [d for d in DATASETS if '3d' not in d]
DATASETS_3D = [d for d in DATASETS if '3d' in d]

# Model configurations
MODELS = {
    '2d': ['cnn', 'resnet18', 'resnet34'],
    '3d': ['cnn', 'resnet18', 'resnet34']
}

@app.route('/')
def index():
    return render_template('index.html', 
                         datasets_2d=DATASETS_2D,
                         datasets_3d=DATASETS_3D)

@app.route('/get_models')
def get_models():
    dataset = request.args.get('dataset', '')
    is_3d = '3d' in dataset
    model_type = '3d' if is_3d else '2d'
    return jsonify({
        'models': MODELS[model_type],
        'is_3d': is_3d
    })

@app.route('/get_saved_models')
def get_saved_models():
    dataset = request.args.get('dataset', '')
    models_dir = os.path.join('saved_models', dataset)
    if not os.path.exists(models_dir):
        return jsonify({'models': []})
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return jsonify({'models': models})

def training_progress_callback(epoch, total_epochs, train_acc, test_acc, lr, train_loss, val_loss):
    """Callback function to send training progress to frontend"""
    progress = {
        'epoch': epoch,
        'total_epochs': total_epochs,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'lr': lr,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    socketio.emit('training_progress', progress)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    dataset = data['dataset']
    model_name = data['model']
    batch_size = int(data['batch_size'])
    num_epochs = int(data['num_epochs'])
    learning_rate = float(data['learning_rate'])
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get dataset info
        info = INFO[dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        # Check if it's a 3D dataset
        is_3d = dataset in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                           'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
        
        # Ensure dataset directory exists
        data_dir = os.path.join('data', dataset)
        os.makedirs(data_dir, exist_ok=True)
        
        # Set environment variable
        os.environ['MEDMNIST_DATASET_FOLDER'] = data_dir
        
        # Check if dataset file exists
        dataset_file = os.path.join(data_dir, f'{dataset}.npz')
        if not os.path.exists(dataset_file):
            print(f"Downloading {dataset} dataset...")
            try:
                DataClass = getattr(medmnist, info['python_class'])
                _ = DataClass(split='train', download=True, root=data_dir)
                print(f"Dataset {dataset} downloaded successfully")
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Failed to download dataset: {str(e)}'})
        
        # Get dataset
        from utils.data_utils import get_medmnist_dataset
        train_dataset, val_dataset, test_dataset, _ = get_medmnist_dataset(dataset)
        
        # Create data loaders
        from torch.utils import data
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Initialize model
        model = get_model(model_name, n_channels, n_classes, is_3d).to(device)
        
        # Loss function and optimizer
        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Train model with progress callback
        train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, 
                   device, task, dataset, model_name, scheduler, 
                   progress_callback=training_progress_callback)
        
        return jsonify({'status': 'success', 'message': 'Training completed!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    model_path = request.form['model_path']
    dataset = request.form['dataset']
    
    try:
        # Save uploaded file
        image_path = os.path.join('web_app', 'static', 'uploads', file.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        file.save(image_path)
        
        # Extract model name from model path (e.g., 'saved_models/pathmnist/best_model_pathmnist_cnn.pth' -> 'cnn')
        model_name = os.path.basename(model_path).split('_')[-1].split('.')[0]
        
        # Make prediction
        prediction, probabilities = predict_image(model_path, image_path, dataset, model_name)
        
        # Get top 3 predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_predictions = [(INFO[dataset]['label'][str(i)], float(probabilities[i])) 
                          for i in top3_idx]
        
        return jsonify({
            'status': 'success',
            'predictions': top3_predictions,
            'image_path': f'/static/uploads/{file.filename}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True) 