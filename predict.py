import torch
import torch.nn as nn
from models.simple_cnn import Net
from models.resnet import ResNet18, ResNet34
from medmnist import INFO
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_utils import get_medmnist_dataset
import torchvision.utils as vutils

def get_model(model_name, in_channels, num_classes):
    """
    Get model based on name
    """
    if model_name == 'cnn':
        return Net(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'resnet18':
        return ResNet18(in_channels=in_channels, num_classes=num_classes)
    elif model_name == 'resnet34':
        return ResNet34(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def save_sample_image(dataset, save_path):
    """
    Save a sample image from the dataset
    """
    # Get a random sample
    sample_idx = np.random.randint(len(dataset))
    sample_image, sample_label = dataset[sample_idx]
    
    # Convert tensor to PIL Image
    if sample_image.shape[0] == 1:  # Grayscale
        sample_image = sample_image.squeeze()
    else:  # RGB
        sample_image = sample_image.permute(1, 2, 0)
    
    # Convert to numpy array and scale to 0-255
    sample_image = (sample_image.numpy() * 255).astype(np.uint8)
    
    # Create PIL Image and save
    if len(sample_image.shape) == 2:  # Grayscale
        image = Image.fromarray(sample_image, mode='L')
    else:  # RGB
        image = Image.fromarray(sample_image)
    
    image.save(save_path)
    return sample_label

def predict_image(model_path, image_path, data_flag, model_name, device='cuda'):
    """
    Predict class for a single image
    """
    # Get dataset info
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L' if n_channels == 1 else 'RGB')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Load model
    model = get_model(model_name, n_channels, n_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        
        if task == 'multi-label, binary-class':
            predicted = (outputs > 0).float()
            probabilities = torch.sigmoid(outputs)
        else:
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
    
    # Get prediction details
    if task == 'multi-label, binary-class':
        predicted_classes = [info['label'][i] for i in range(n_classes) if predicted[0][i] == 1]
        probabilities = probabilities[0].cpu().numpy()
    else:
        predicted_class = info['label'][str(predicted.item())]  # Convert to string for dictionary key
        probabilities = probabilities[0].cpu().numpy()
    
    # Visualize prediction
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray' if n_channels == 1 else None)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot probability distribution
    plt.subplot(1, 2, 2)
    plt.bar(range(n_classes), probabilities)
    plt.title('Class Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(range(n_classes), [info['label'][str(i)] for i in range(n_classes)], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'prediction_{model_name}.png')
    plt.close()
    
    # Save prediction details to file
    with open(f'prediction_details_{model_name}.txt', 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Dataset: {data_flag}\n')
        f.write(f'Task: {task}\n\n')
        
        if task == 'multi-label, binary-class':
            f.write('Predicted Classes:\n')
            for class_name in predicted_classes:
                f.write(f'- {class_name}\n')
        else:
            f.write(f'Predicted Class: {predicted_class}\n')
        
        f.write('\nProbabilities:\n')
        for i in range(n_classes):
            f.write(f'{info["label"][str(i)]}: {probabilities[i]:.4f}\n')
    
    return predicted_classes if task == 'multi-label, binary-class' else predicted_class, probabilities

if __name__ == '__main__':
    # Configuration
    data_flag = 'pathmnist'  # Change this to use different MedMNIST datasets
    model_name = 'resnet18'  # Choose from: 'cnn', 'resnet18', 'resnet34'
    model_path = f'best_model_{model_name}.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running 'python run.py'")
        exit(1)
    
    # Get dataset and save a sample image
    _, _, test_dataset, _ = get_medmnist_dataset(data_flag)
    sample_label = save_sample_image(test_dataset, 'sample_image.png')
    
    # Make prediction
    prediction, probabilities = predict_image(model_path, 'sample_image.png', data_flag, model_name, device)
    
    # Print results
    print(f'\nPrediction Results for {model_name}:')
    print(f'True Label: {INFO[data_flag]["label"][str(sample_label.item())]}')
    if isinstance(prediction, list):
        print('Predicted Classes:')
        for class_name in prediction:
            print(f'- {class_name}')
    else:
        print(f'Predicted Class: {prediction}')
    
    print('\nProbabilities:')
    info = INFO[data_flag]
    for i in range(len(info['label'])):
        print(f'{info["label"][str(i)]}: {probabilities[i]:.4f}') 