import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(dataset_name='digits', test_size=0.2, random_state=42):
    if dataset_name == 'digits':
        digits = load_digits()
        X = digits.data
        y = digits.target
    elif dataset_name == 'wine':
        wine = datasets.load_wine()
        X = wine.data
        y = wine.target
    elif dataset_name == 'moons':
        X, y = make_moons(n_samples=3200, noise=0.1, random_state=42)
    
    # Standardize the features
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