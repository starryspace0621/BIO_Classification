import torch
import torch.nn as nn
from models.simple_cnn import Net, Net3D
from models.resnet import ResNet18, ResNet34, ResNet183D, ResNet343D
from medmnist import INFO
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_utils import get_medmnist_dataset
import torchvision.utils as vutils
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

def save_sample_image(dataset, save_path, is_3d=False):
    """
    Save a sample image from the dataset
    """
    # Get a random sample
    sample_idx = np.random.randint(len(dataset))
    sample_image, sample_label = dataset[sample_idx]
    
    if is_3d:
        # For 3D images, save a middle slice
        middle_slice = sample_image.shape[-1] // 2
        sample_image = sample_image[..., middle_slice]
    
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
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    
    # 创建结果保存目录
    results_dir = os.path.join('results', data_flag, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if the dataset is 3D
    is_3d = data_flag in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                         'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
    
    # Load and preprocess image
    if is_3d:
        # For 3D images, we need to load the entire volume
        try:
            image = np.load(image_path, allow_pickle=True)
        except:
            # 如果直接加载失败，尝试从数据集获取一个样本
            _, _, test_dataset, _ = get_medmnist_dataset(data_flag)
            sample_idx = np.random.randint(len(test_dataset))
            image, _ = test_dataset[sample_idx]
            if isinstance(image, torch.Tensor):
                image = image.numpy()
    else:
        image = Image.open(image_path).convert('L' if n_channels == 1 else 'RGB')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 对于3D数据，确保数据格式正确
    if is_3d:
        if isinstance(image, np.ndarray):
            # 确保数据是float类型
            image = image.astype(np.float32)
            # 添加batch维度
            image = np.expand_dims(image, axis=0)
            # 转换为tensor
            image_tensor = torch.from_numpy(image).to(device)
        else:
            raise TypeError(f"Unsupported image type for 3D data: {type(image)}")
    
    # Load model
    model = get_model(model_name, n_channels, n_classes, is_3d).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        
        if task == 'multi-label, binary-class':
            predicted = (torch.sigmoid(outputs) > 0.5).float()
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
    if is_3d:
        # For 3D images, show the middle slice
        middle_slice = image.shape[-1] // 2
        # 调整维度顺序：从(C, H, W, D)变为(H, W)
        if len(image.shape) == 4:  # (B, C, H, W, D)
            slice_image = image[0, 0, :, :, middle_slice]  # 取第一个batch，第一个通道
        else:  # (C, H, W, D)
            slice_image = image[0, :, :, middle_slice]  # 取第一个通道
        # 确保图像是2D的
        if len(slice_image.shape) == 3:
            slice_image = slice_image.squeeze()
        plt.imshow(slice_image, cmap='gray' if n_channels == 1 else None)
    else:
        # 对于2D图像，确保维度正确
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
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
    plt.savefig(os.path.join(results_dir, f'prediction_{data_flag}_{model_name}.png'))
    plt.close()
    
    # Save prediction details to file
    with open(os.path.join(results_dir, f'prediction_details_{data_flag}_{model_name}.txt'), 'w') as f:
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
    
    # Ensure the dataset directory exists
    data_dir = os.path.join('data', data_flag)
    os.makedirs(data_dir, exist_ok=True)
    
    # Set environment variable to force MedMNIST to use our data directory
    os.environ['MEDMNIST_DATASET_FOLDER'] = data_dir
    
    # Check if dataset file exists
    dataset_file = os.path.join(data_dir, f'{data_flag}.npz')
    if not os.path.exists(dataset_file):
        print(f"Downloading {data_flag} dataset...")
        try:
            # Attempt to download the dataset
            DataClass = getattr(medmnist, info['python_class'])
            _ = DataClass(split='train', download=True, root=data_dir)
            print(f"Dataset {data_flag} downloaded successfully")
        except Exception as e:
            print(f"Error while downloading dataset: {str(e)}")
            print("Please check your network connection or try downloading the dataset manually")
            exit(1)
    
    # Check if the dataset is 3D
    is_3d = data_flag in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                         'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
    
    # 创建结果目录
    results_dir = os.path.join('results', data_flag, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get dataset and save a sample image
    _, _, test_dataset, _ = get_medmnist_dataset(data_flag)
    sample_label = save_sample_image(test_dataset, 
                                   os.path.join(results_dir, f'sample_image_{data_flag}.png'), 
                                   is_3d)
    
    # Make prediction
    prediction, probabilities = predict_image(model_path, 
                                           os.path.join(results_dir, f'sample_image_{data_flag}.png'), 
                                           data_flag, model_name, device)
    
    # Print results
    print(f'\nPrediction Results ({data_flag}, {model_name}):')
    
    # Process the true label
    if info['task'] == 'multi-label, binary-class':
        true_labels = [info['label'][i] for i in range(len(info['label'])) if sample_label[i] == 1]
        print('True Labels:')
        for label in true_labels:
            print(f'- {label}')
    else:
        print(f'True Label: {info["label"][str(sample_label.item())]}')
    
    if isinstance(prediction, list):
        print('Predicted Classes:')
        for class_name in prediction:
            print(f'- {class_name}')
    else:
        print(f'Predicted Class: {prediction}')
    
    print('\nProbability Distribution:')
    for i in range(len(info['label'])):
        print(f'{info["label"][str(i)]}: {probabilities[i]:.4f}')