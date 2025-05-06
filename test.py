import torch
import torch.nn as nn
from torch.utils import data
from models.simple_cnn import Net, Net3D
from models.resnet import ResNet18, ResNet34, ResNet183D, ResNet343D
from utils.data_utils import get_medmnist_dataset
from medmnist import INFO, Evaluator
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import medmnist

def get_model(model_name, in_channels, num_classes, is_3d=False):
    """
    Get model based on name
    """
    if is_3d:
        if model_name == 'cnn':
            return Net3D(in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'resnet18':
            return ResNet183D(in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'resnet34':
            return ResNet343D(in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    else:
        if model_name == 'cnn':
            return Net(in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'resnet18':
            return ResNet18(in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'resnet34':
            return ResNet34(in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

def test_model(model, test_loader, criterion, device, data_flag, model_name):
    """
    Test the model
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # 创建结果保存目录
    results_dir = os.path.join('results', data_flag, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 计算并保存测试结果
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    # 保存测试结果到文件
    with open(os.path.join(results_dir, f'test_results_{data_flag}_{model_name}.txt'), 'w') as f:
        f.write(f'Test Results for {model_name} on {data_flag}\n')
        f.write(f'Average Loss: {avg_loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.2f}%\n')
        f.write(f'Correct/Total: {correct}/{total}\n')
    
    # 计算并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} on {data_flag}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{data_flag}_{model_name}.png'))
    plt.close()
    
    return accuracy, avg_loss

if __name__ == '__main__':
    # Add command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, default='organmnist3d',
                      help='Dataset name, e.g., pathmnist, chestmnist, etc.')
    parser.add_argument('--model_name', type=str, default='cnn',
                      help='Model name: cnn, resnet18, resnet34')
    args = parser.parse_args()
    
    # Configuration
    data_flag = args.data_flag
    model_name = args.model_name
    model_path = f'saved_models/{data_flag}/best_model_{data_flag}_{model_name}.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist. Please train the model first.")
        exit(1)
    
    # Check if dataset exists
    try:
        info = INFO[data_flag]
    except KeyError:
        print(f"Error: Dataset {data_flag} does not exist. Available datasets:")
        print(list(INFO.keys()))
        exit(1)
    
    # Get test dataset
    _, _, test_dataset, _ = get_medmnist_dataset(data_flag)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    # Load model
    model = get_model(model_name, info['n_channels'], len(info['label']), data_flag in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                         'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d'])
    model.load_state_dict(torch.load(model_path))
    
    # 确保数据集目录存在
    data_dir = os.path.join('data', data_flag)
    os.makedirs(data_dir, exist_ok=True)
    
    # 设置环境变量，强制MedMNIST使用我们的数据目录
    os.environ['MEDMNIST_DATASET_FOLDER'] = data_dir
    
    # 创建结果目录
    results_dir = os.path.join('results', data_flag, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Test the model
    accuracy, avg_loss = test_model(model, test_loader, nn.CrossEntropyLoss(), device, data_flag, model_name)
    
    print(f'Test Results ({data_flag}, {model_name}):')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')