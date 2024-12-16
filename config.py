import numpy as np

# Define the base class for activation functions
class ActivationFunction:
    def rho(self, x):
        raise NotImplementedError("Forward method not implemented")
    
    def rhop(self, x):
        raise NotImplementedError("Derivative method not implemented")

# Sigmoid1 Activation Function
class Sigmoid1(ActivationFunction):
    def rho(self, x):
        return 1 / (1 + np.exp(-(4 * (x - 0.5))))
    
    def rhop(self, x):
        rho_x = self.rho(x)
        return 4 * rho_x * (1 - rho_x)

# Sigmoid2 Activation Function
class Sigmoid2(ActivationFunction):
    def rho(self, x):
        return 1 / (1 + np.exp(-x))
    
    def rhop(self, x):
        rho_x = self.rho(x)
        return rho_x * (1 - rho_x)

# Hard Sigmoid Activation Function
class HardSigmoid(ActivationFunction):
    def rho(self, x):
        return np.clip(x, 0, 1)
    
    def rhop(self, x):
        return (x >= 0) & (x <= 1)

# Tanh Activation Function
class Tanh(ActivationFunction):
    def rho(self, x):
        return np.tanh(x)
    
    def rhop(self, x):
        return 1 - np.power(np.tanh(x), 2)

activation_dict = {
    "Sigmoid1": Sigmoid1(),
    "Sigmoid2": Sigmoid2(),
    "HardSigmoid": HardSigmoid(),
    "Tanh": Tanh()
}



simulation_params = {
    'digits_simulation': {
        'dataset_name': 'digits',
        'epochs': 100,
        'batch_size': 12,
        'test_size': 0.2,
        'random_state': 42,
        'fcLayers': [10, 128, 64],
        'dt': 0.5,
        'T': 10,
        'Kmax': 5,
        'beta': 1,
        'loss': 'MSE',
        'activation_function': activation_dict['Tanh'],
        'lr_first': 0.01,
        'lr_second': 0.03,
        'data': '/path/to/memristor_data'
    },
    'wine_simulation': {
        'dataset_name': 'wine',
        'epochs': 50,
        'batch_size': 32,
        'test_size': 0.25,
        'random_state': 42,
        'fcLayers': [3, 100, 13],
        'dt': 0.5,
        'T': 12,
        'Kmax': 4,
        'beta': 0.9,
        'loss': 'Cross-entropy',
        'activation_function': activation_dict['Tanh'],
        'lr_first': 0.01,
        'lr_second': 0.03,
        'data': '/path/to/memristor_data'
    },
    'mnist_simulation': {
        'dataset_name': 'mnist',
        'epochs': 50,
        'batch_size': 1024,
        'test_size': 0.2,
        'random_state': 42,
        'fcLayers': [10, 512, 784],  # Input (28x28 flattened), hidden layers, output (10 classes)
        'dt': 0.5,
        'T': 15,
        'Kmax': 6,
        'beta': 0.4,
        'loss': 'MSE',
        'activation_function': activation_dict['Tanh'],
        'lr_first': 0.02,
        'lr_second': 0.06,
        'data': '/path/to/memristor_data'
    }
}



