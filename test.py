import torch
import torch.nn as nn
from torch.utils import data
from models.simple_cnn import Net
from models.resnet import ResNet18, ResNet34
from utils.data_utils import get_medmnist_dataset
from medmnist import INFO, Evaluator
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

def test(model_path, data_flag, model_name, device='cuda'):
    """
    Test the model
    """
    # Get dataset info
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Get test dataset
    _, _, test_dataset, _ = get_medmnist_dataset(data_flag)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    # Load model
    model = get_model(model_name, n_channels, n_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Test the model
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                predicted = (outputs > 0).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    if task == 'multi-label, binary-class':
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='samples')
        recall = recall_score(all_labels, all_preds, average='samples')
        f1 = f1_score(all_labels, all_preds, average='samples')
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Create confusion matrix
    if task != 'multi-label, binary-class':
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()
    
    # Save metrics to file
    with open(f'test_metrics_{model_name}.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    # Configuration
    data_flag = 'pathmnist'  # Change this to use different MedMNIST datasets
    model_name = 'resnet18'  # Choose from: 'cnn', 'resnet18', 'resnet34'
    model_path = f'best_model_{model_name}.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test the model
    accuracy, precision, recall, f1 = test(model_path, data_flag, model_name, device)
    
    print(f'Test Results for {model_name}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}') 