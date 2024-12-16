import torch
from sklearn.datasets import load_digits, load_wine, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def load_and_preprocess_data(dataset_name='digits', test_size=0.2, random_state=42):
    if dataset_name == 'digits':
        digits = load_digits()
        X = digits.data
        y = digits.target
    elif dataset_name == 'wine':
        wine = load_wine()
        X = wine.data
        y = wine.target
    elif dataset_name == 'moons':
        X, y = make_moons(n_samples=3200, noise=0.1, random_state=42)
    elif dataset_name == 'mnist':
        # Load MNIST dataset
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        
        # Flatten the images and normalize manually (MNIST is already normalized by torchvision)
        X_train = mnist_train.data.view(-1, 28 * 28).float() / 255.0
        y_train = mnist_train.targets
        X_test = mnist_test.data.view(-1, 28 * 28).float() / 255.0
        y_test = mnist_test.targets
        
        # Create TensorDataset objects
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    
    # Standardize the features (skip for MNIST as it is handled separately above)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=test_size, random_state=random_state)
    
    # Create TensorDataset objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    return train_dataset, test_dataset

def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader