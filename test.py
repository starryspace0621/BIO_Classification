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
import argparse
import medmnist

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
    # 确保数据集目录存在
    data_dir = os.path.join('data', data_flag)
    os.makedirs(data_dir, exist_ok=True)
    
    # 设置环境变量，强制MedMNIST使用我们的数据目录
    os.environ['MEDMNIST_DATASET_FOLDER'] = data_dir
    
    # 检查数据集文件是否存在
    dataset_file = os.path.join(data_dir, f'{data_flag}.npz')
    if not os.path.exists(dataset_file):
        print(f"正在下载 {data_flag} 数据集...")
        try:
            # 尝试直接下载数据集
            DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
            _ = DataClass(split='train', download=True, root=data_dir)
            print(f"数据集 {data_flag} 下载完成")
        except Exception as e:
            print(f"下载数据集时出错: {str(e)}")
            print("请检查网络连接或尝试手动下载数据集")
            return None, None, None, None
    
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
                # 使用sigmoid激活函数并设置阈值
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    if task == 'multi-label, binary-class':
        # 对于多标签分类，使用不同的评估方式
        accuracy = accuracy_score(all_labels, all_preds)
        # 设置zero_division参数避免警告
        precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Create confusion matrix
    if task != 'multi-label, binary-class':
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({data_flag}, {model_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{data_flag}_{model_name}.png')
        plt.close()
    
    # Save metrics to file
    with open(f'test_metrics_{data_flag}_{model_name}.txt', 'w') as f:
        f.write(f'Dataset: {data_flag}\n')
        f.write(f'Model: {model_name}\n')
        f.write(f'Task: {task}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    # Add command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, default='pathmnist',
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
    
    # Test the model
    accuracy, precision, recall, f1 = test(model_path, data_flag, model_name, device)
    
    print(f'Test Results ({data_flag}, {model_name}):')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')