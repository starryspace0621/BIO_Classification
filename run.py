import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from models.simple_cnn import Net
from models.resnet import ResNet18, ResNet34
from utils.data_utils import get_medmnist_dataset
from medmnist import INFO, Evaluator
from tqdm import tqdm
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

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, task, data_flag, model_name):
    """
    Train the model
    """
    best_acc = 0.0
    
    # Set the data directory
    data_dir = os.path.join('data', data_flag)
    os.environ['MEDMNIST_DATASET_FOLDER'] = data_dir
    
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        
        # Training phase
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                # For multi-label classification, use the sigmoid activation function
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.numel()  # Use the total number of elements instead of the number of samples
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
            
            loss.backward()
            optimizer.step()
        
        # Calculate the training accuracy
        train_acc = 100 * train_correct / train_total
        
        # Testing phase
        model.eval()
        y_true = torch.tensor([], device=device)
        y_score = torch.tensor([], device=device)
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = torch.sigmoid(outputs)  # 使用sigmoid激活函数
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
        
        # Move tensors to CPU for evaluation
        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()
        
        # Ensure the dataset file exists
        dataset_file = os.path.join(data_dir, f'{data_flag}.npz')
        if not os.path.exists(dataset_file):
            print(f"Downloading {data_flag} dataset...")
            try:
                DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
                _ = DataClass(split='train', download=True, root=data_dir)
                print(f"Dataset {data_flag} downloaded successfully")
            except Exception as e:
                print(f"Error while downloading dataset: {str(e)}")
                print("Please check your network connection or try downloading the dataset manually")
                return
        
        # Specify the data directory when creating the Evaluator
        evaluator = Evaluator(data_flag, 'test', root=data_dir)
        metrics = evaluator.evaluate(y_score)
        test_acc = metrics[1] * 100  # Convert to percentage
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Testing Accuracy: {test_acc:.2f}%')

        # Define save directory
        save_dir = os.path.join('saved_models', data_flag)

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'saved_models/{data_flag}/best_model_{data_flag}_{model_name}.pth')

def main():
    # Add command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, default='chestmnist', 
                      help='Dataset name, e.g., pathmnist, chestmnist, etc.')
    parser.add_argument('--model_name', type=str, default='cnn',
                      help='Model name: cnn, resnet18, resnet34')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    data_flag = args.data_flag
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    # Check if dataset exists, otherwise download it
    try:
        info = INFO[data_flag]
    except KeyError:
        print(f"Error: Dataset {data_flag} does not exist. Available datasets:")
        print(list(INFO.keys()))
        return
    
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
            return
    
    # Get dataset info
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Get datasets
    train_dataset, val_dataset, test_dataset, _ = get_medmnist_dataset(data_flag)
    
    # Create data loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = get_model(model_name, n_channels, n_classes).to(device)
    
    # Loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, task, data_flag, model_name)

if __name__ == '__main__':
    main()