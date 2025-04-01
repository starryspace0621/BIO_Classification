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
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            if task == 'multi-label, binary-class':
                predicted = (outputs > 0).float()
                train_correct += (predicted == targets).sum().item()
            else:
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
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
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
        
        # Move tensors to CPU for evaluation
        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()
        
        evaluator = Evaluator(data_flag, 'test')
        metrics = evaluator.evaluate(y_score)
        test_acc = metrics[1] * 100  # Convert to percentage
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Testing Accuracy: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    data_flag = 'pathmnist'  # Change this to use different MedMNIST datasets
    model_name = 'cnn'  # Choose from: 'cnn', 'resnet18', 'resnet34'
    batch_size = 128
    num_epochs = 3
    learning_rate = 0.001
    
    # Get dataset info
    info = INFO[data_flag]
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