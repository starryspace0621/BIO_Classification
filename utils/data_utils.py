import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import medmnist
from medmnist import INFO, Evaluator
import numpy as np

class MedicalImageDataset(Dataset):
    """
    Custom dataset for medical images
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms():
    """
    Get data transforms for training and validation
    """
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform 

def get_medmnist_dataset(data_flag, download=True):
    """
    Get MedMNIST dataset
    Args:
        data_flag: str, name of the dataset (e.g., 'pathmnist', 'chestmnist')
        download: bool, whether to download the dataset if not found
    """
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Create data directory
    data_dir = os.path.join('data', data_flag)
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if dataset exists
    dataset_exists = os.path.exists(os.path.join(data_dir, f'{data_flag}.npz'))
    
    # Data transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load datasets
    if dataset_exists:
        print(f"Loading {data_flag} dataset from {data_dir}")
        download = False
    else:
        print(f"Downloading {data_flag} dataset to {data_dir}")
    
    # 设置环境变量，强制MedMNIST使用我们的数据目录
    os.environ['MEDMNIST_DATASET_FOLDER'] = data_dir
    
    train_dataset = DataClass(split='train', transform=data_transform, download=download, root=data_dir)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, root=data_dir)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, root=data_dir)
    
    # Get number of classes from the dataset
    n_classes = len(info['label'])
    
    # Print dataset information
    print(f"Dataset: {data_flag}")
    print(f"Number of classes: {n_classes}")
    print(f"Task type: {info['task']}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, n_classes

def get_data_loaders(data_flag, batch_size=32, download=True):
    """
    Get data loaders for MedMNIST dataset
    """
    train_dataset, val_dataset, test_dataset, num_classes = get_medmnist_dataset(data_flag, download)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes

def save_medmnist_sample(data_flag, save_dir='data/samples', num_samples=5):
    """
    Save sample images from MedMNIST dataset
    """
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Create data directory
    data_dir = os.path.join('data', data_flag)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if dataset exists
    dataset_exists = os.path.exists(os.path.join(data_dir, f'{data_flag}_train.npz'))
    
    # Load dataset without transforms for visualization
    if dataset_exists:
        print(f"Loading {data_flag} dataset from {data_dir}")
        dataset = DataClass(split='train', download=False, root=data_dir)
    else:
        print(f"Downloading {data_flag} dataset to {data_dir}")
        dataset = DataClass(split='train', download=True, root=data_dir)
    
    # Save samples
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        
        # Convert to PIL Image if it's not already
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()
        elif isinstance(img, np.ndarray):
            img = img.squeeze()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        
        # Normalize to 0-255 range
        img = ((img + 0.5) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        
        # Save image
        img.save(os.path.join(save_dir, f'sample_{i}_class_{label}.png')) 