import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
import optuna
from plots_eq_prop import *
from optuna.trial import TrialState
from sklearn.preprocessing import StandardScaler
import os


def one_hot_encode(labels, num_classes):
    # Create a tensor of zeros with shape (len(labels), num_classes)
    target = torch.zeros((len(labels), num_classes))
    
    # Use scatter_ to assign 1s in the correct class positions
    target.scatter_(1, labels.unsqueeze(1), 1)
    return target



def plot_accuracy(accuracy_list, act_fn_name):
    """
    Plots the accuracy list for each epoch and includes the activation function name in the plot title.
    
    Parameters:
    - accuracy_list: List of accuracy values for each epoch.
    - act_fn_name: Name of the activation function used in the model.
    """
    epochs = range(1, len(accuracy_list) + 1)  # Create epoch numbers
    
    # Plot the accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy_list, marker='o', linestyle='-', label='Accuracy')
    plt.title(f'Accuracy per Epoch (Activation: {act_fn_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    
    # Save the plot as a file with act_fn_name in the filename
    filename = f'accuracy_plot_{act_fn_name}.png'
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    
    # Show the plot
    plt.show()












# Create the network
class mlp_eqprop(nn.Module):
    def __init__(self, fcLayers, dt, T, Kmax, beta, loss, act_fn):
        super(mlp_eqprop, self).__init__()
        self.fcLayers = fcLayers
        self.dt = dt
        self.T = T
        self.Kmax = Kmax
        self.beta = beta
        self.loss = loss
        self.act_fn = act_fn
        if loss == 'MSE':
            self.softmax_output = False
        elif loss == 'Cross-entropy':
            self.softmax_output = True
            
        W = nn.ModuleList(None)
        Wp = nn.ModuleList(None)
        Wn = nn.ModuleList(None)
        
        for i in range(len(fcLayers)-1):
            W.extend([nn.Linear(fcLayers[i+1], fcLayers[i], bias=True)])
            Wp.extend([nn.Linear(fcLayers[i+1], fcLayers[i], bias=False)])
            Wn.extend([nn.Linear(fcLayers[i+1], fcLayers[i], bias=False)])
            
        self.W = W
        self.Wp = Wp
        self.Wn = Wn




    #softmax readout of the last layer


    def stepper_softmax(self, s, target=None, beta= None): #Do I have here actually one layer less - the softmax layer is added on top of the last layer?

        if len(s) < 3:
            raise ValueError("Input list 's' must haave at least three elements for softmax-readout.")
    
        # Separate 'h' elements and 'y'
        h = s[1:]  # All but the first element are considered 'h' # at the start s has only the inputs
        y = F.softmax(self.W[0](self.act_fn.rho(h[0])), dim=1) #softmax for the last layer
    
        dhdt = [-h[0] + self.act_fn.rhop(h[0]) *self.W[1](self.act_fn.rho(h[1]))] #the update for the first hidden layer from the input
        
        if target is not None and beta is not None:
            dhdt[0] = dhdt[0] + beta * torch.mm((target-y), self.W[0].weight) #nudge of the output layer - softmax layer doesn't propagate
    
        for layer in range(1, len(h) - 1):
            dhdt.append(-h[layer] + self.act_fn.rhop(h[layer]) * (self.W[layer+1](self.act_fn.rho(h[layer+1]))
                                                               + torch.mm(self.act_fn.rho(h[layer - 1]),self.W[layer].weight)))
        # update h
        for (layer, dhdt_item) in enumerate(dhdt):
                h[layer] = h[layer] + self.dt * dhdt_item
                h[layer] = h[layer].clamp(0, 1)
            
        return [y] + h
        
    def stepper_c(self, s, target=None, beta=None): #original
            """
            stepper function for energy-based dynamics of EP
            """
            if len(s) < 2:
                raise ValueError("Input list 's' must have at least two elements.")
            #this is dsdt for the output layer
            dsdt = [-s[0] + (self.act_fn.rhop(s[0])*(self.W[0](self.act_fn.rho(s[1]))))]
            #in the nudge phase the output layer is clamped
            if beta is not None and target is not None:
                dsdt[0] = dsdt[0] + beta*(target-s[0])
                    #here calculate the updates layer by layer
            for layer in range(1, len(s)-1):  # start at the first hidden layer and then to the before last hidden layer
                dsdt.append(-s[layer] + self.act_fn.rhop(s[layer])*(self.W[layer](self.act_fn.rho(s[layer+1])) + torch.mm(self.act_fn.rho(s[layer-1]), self.W[layer-1].weight)))
    
            for (layer, dsdt_item) in enumerate(dsdt):
                s[layer] = s[layer] + self.dt*dsdt_item
                s[layer] = s[layer].clamp(0, 1)
    
            return s


    def stepper_mixed(self, s, target=None, beta=None): #one hidden layer one linear layer
            """
            stepper function for energy-based dynamics of EP
            """
            if len(s) < 2:
                raise ValueError("Input list 's' must have at least two elements.")
            #this is dsdt for the output layer
            
            #rho2 = 1/(1+np.exp(-(4*(x-0.5))))
            
            dsdt = [-s[0] + s[0]*(self.W[0](self.act_fn.rho(s[1])))]
            #in the nudge phase the output layer is clamped
            if beta is not None and target is not None:
                dsdt[0] = dsdt[0] + beta*(target-s[0])
                    #here calculate the updates layer by layer
             #start at the first hidden layer and then to the before last hidden layer
            dsdt.append((-s[1] + self.act_fn.rhop(s[1])*(self.W[1](self.act_fn.rho(s[2])) + torch.mm(s[0], self.W[0].weight))))
            #dsdt[2](-s[2] + rhop(s[2])*(self.W[1]((s[1+1])) + torch.mm(rho(s[1-1]), self.W[1-1].weight)))

            for (layer, dsdt_item) in enumerate(dsdt):
                s[layer] = s[layer] + self.dt*dsdt_item
                s[layer] = s[layer].clamp(0, 1)
    
            return s





    # def stepper_softmax(self, s, target=None, beta= None):

    #     if len(s) < 3:
    #         raise ValueError("Input list 's' must haave at least three elements for softmax-readout.")
    
    #     # Separate 'h' elements and 'y'
    #     h = s[1:]  # All but the first element are considered 'h' # at the start s has only the inputs
    #     y = F.softmax(self.W[0](rho(h[0])), dim=1) #softmax for the last layer
    
    #     dhdt = [-h[0] + rhop(h[0]) *self.W[1](rho(h[1]))] #the update for the first hidden layer from the input
        
    #     if target is not None and beta is not None:
    #         dhdt[0] = dhdt[0] + beta * torch.mm((target-y), self.W[0].weight) #nudge of the output layer
    
    #     for layer in range(1, len(h) - 1):
    #         dhdt.append(-h[layer] + rhop(h[layer]) * (self.W[layer+1](rho(h[layer+1]))
    #                                                            + torch.mm(rho(h[layer - 1]),self.W[layer].weight)))
    #     # update h
    #     for (layer, dhdt_item) in enumerate(dhdt):
    #             h[layer] = h[layer] + self.dt * dhdt_item
    #             h[layer] = h[layer].clamp(0, 1)
            
    #     return [y] + h




    
    def forward(self, s, beta=None, target=None, tracking=False):
        #update parameters for all time steps
        T, Kmax = self.T, self.Kmax
        if beta is None and target is None:
            q, y = torch.empty((s[1].size(1),T)), torch.empty((s[0].size(1), T))

        else:
            q, y = torch.empty((s[1].size(1),Kmax)), torch.empty((s[0].size(1), Kmax))

        with torch.no_grad():
            # continuous time EP
            if beta is None and target is None:
                # free phase
                if self.softmax_output:
                    for t in range(T):
                        s = self.stepper_softmax(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]
                else:
                    for t in range(T):
                        s = self.stepper_c(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]
            else:
                # nudged phase
                if self.softmax_output:
                    for t in range(Kmax):
                        s = self.stepper_softmax(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]
                else:
                    for t in range(Kmax):
                        s = self.stepper_c(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]

        return s, q, y


    def compute_gradients_ep(self, s, seq, target=None):
        """
        Compute EQ gradient to update the synaptic weight
        """
        batch_size = s[0].size(0)
        coef = 1 / (self.beta * batch_size)
        gradW, gradBias = [], []

        with torch.no_grad():
            if self.softmax_output:
                gradW.append(
                    -(0.5 / batch_size) * (torch.mm(torch.transpose((s[0] - target), 0, 1), self.act_fn.rho(s[1])) +
                                           torch.mm(torch.transpose((seq[0] - target), 0, 1),
                                                    self.act_fn.rho(seq[1]))))
                gradBias.append(-(0.5 / batch_size) * (self.s[0] + seq[0] - 2 * target).sum(0))
            else:
                gradW.append(coef * (torch.mm(torch.transpose(self.act_fn.rho(s[0]), 0, 1), self.act_fn.rho(s[1])) -
                                     torch.mm(torch.transpose(self.act_fn.rho(seq[0]), 0, 1),
                                              self.act_fn.rho(seq[1]))))
                gradBias.append(coef * (self.act_fn.rho(s[0]) - self.act_fn.rho(seq[0])).sum(0))

            for layer in range(1, len(s) - 1):
                gradW.append(coef * (torch.mm(torch.transpose(self.act_fn.rho(s[layer]), 0, 1), self.act_fn.rho(s[layer+1])) -
                                     torch.mm(torch.transpose(self.act_fn.rho(seq[layer]), 0, 1),
                                              self.act_fn.rho(seq[layer+1]))))
                gradBias.append(coef * (self.act_fn.rho(s[layer]) - self.act_fn.rho(seq[layer])).sum(0))

        for i, param in enumerate(self.W):
            param.weight.grad = -gradW[i]
            param.bias.grad = -gradBias[i]



    
    def init_state(self, data):
        """
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        """
        state = []
        size = data.size(0)
        for layer in range(len(self.fcLayers) - 1): #set everything to zero except the last layer with the inputs
            state.append(torch.zeros(size, self.fcLayers[layer], requires_grad=False))

        state.append(data.float())

        return state




def init_weights(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform_(m.weight)  # Xavier uniform
        # Or Xavier normal
        torch.nn.init.xavier_normal_(m.weight) 

def defineOptimizer(net, lr, type):
    net_params = []
    for i in range(len(net.W)):
        net_params += [{'params': [net.W[i].weight], 'lr': lr[i]}]
        net_params += [{'params': [net.W[i].bias], 'lr': lr[i]}]
    if type == 'SGD':
        optimizer = torch.optim.SGD(net_params)
    elif type == 'Adam':
        optimizer = torch.optim.Adam(net_params)
    else:
        raise ValueError("{} type of Optimizer is not defined ".format(type))

    return net_params, optimizer


    
def objective(trial):    
    # Parameters for the network
    
    
    
    # X, y = make_moons(n_samples=3200, noise=0.1, random_state=42)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # X = scaler.fit_transform(X)
    
    # # Convert to PyTorch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.long)
    
    # # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    # # Create TensorDataset objects
    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)
    
    # # Create DataLoader objects
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    
    
    # wine = datasets.load_wine()
    # X = wine.data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # y = wine.target
    # # scale_low = trial.suggest_float("scale_low", -3, -0.5, log=False)
    # # scale_high = -scale_low
    # # scaler = MinMaxScaler(feature_range=(scale_low, scale_high))
    # # X = scaler.fit_transform(X)
    
    # # Convert to PyTorch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.long)
    
    # # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    # # Create TensorDataset objects
    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)
    
    # # Create DataLoader objects
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    
    # Load the DIGITS dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Create TensorDataset objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    
    
    
    
    
    

    # # Define transforms for data preprocessing (normalization)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),                     # Convert images to PyTorch tensors
    #     transforms.Normalize((0.5,), (0.5,))      # Normalize pixel values to [-1, 1]
    # ])
    
    # # Load the MNIST dataset
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    
    # # Check dataset shape
    # print(f"Training data size: {len(train_dataset)} samples")
    # print(f"Test data size: {len(test_dataset)} samples")
    
    
    # # # Create DataLoader objects
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    
    

    # Visualize the dataset
    
    
    
    
    
    dt = 0.5
    T = 10  # free phase time
    Kmax = 5  # nudge phase time
    fcLayers = [10, 128, 64] # [output, hidden, input]
    loss = 'MSE'  # 'MSE' or 'Cross-entropy' 
    beta = 1
    lr_first = 0.03
    lr_second = 0.09
    lr = [lr_first, lr_second]
    
    # Optuna suggests an activation function name (not the instance directly)
    act_fn_name = trial.suggest_categorical("activation_function", activation_dict.keys())
    # Retrieve the corresponding activation function instance from the dictionary
    act_fn = activation_dict[act_fn_name]

    # Now you can pass the selected activation function to your network
    net = mlp_eqprop(fcLayers, dt, T, Kmax, beta, loss, act_fn)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    net_params, optimizer = defineOptimizer(net, lr, 'SGD')
    net.apply(init_weights)
    epoch = 100
    #criterion = nn.CrossEntropyLoss()
    
    weight_history = []
    neuron_values_hidden = []
    neuron_values_output = []
    targets_array = []
    losses = []
    accuracy_list = []
    epochs_to_plot = [1, 40, 80, 120, 160, 190]
    epoch_weight_snapshots = {}
    
    for rep in range(epoch):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)
            targets = one_hot_encode(targets, 10)
            s = net.init_state(data)
            s, m, n = net.forward(s, tracking=True)
            if batch_idx == 0:
                train_hidden = m
                train_output = n
            else:
                train_hidden = torch.cat((train_hidden, m), 1)
                train_output = torch.cat((train_output, n), 1)
            seq = s.copy()
            s, m, n = net.forward(s, beta=beta, target=targets, tracking=True)
            train_hidden = torch.cat((train_hidden, m), 1)
            train_output = torch.cat((train_output, n), 1)
            # Update weight
            net.compute_gradients_ep(s, seq, targets)
            optimizer.step()
            
            neuron_values_hidden.append(m[:, -1])
            targets_array.append(targets)
            
        # Collect weights for the current epoch
        weight_snapshots = collect_weights(net.W)
    
        # Store weights for specific epochs
        if rep + 1 in epochs_to_plot:
            epoch_weight_snapshots[rep + 1] = weight_snapshots
    
        weight_history.append(weight_snapshots)
        #plot_losses(losses)
        #plot_neuron_outputs(neuron_values_hidden)
        #plot_neuron_statistics(neuron_values_hidden)
        # Test
        test_number = 0
        correct_number = 0
        
        for index, (data, targets) in enumerate(test_loader):
            data = data.view(data.size(0), -1)
            s = net.init_state(data)
            s, m, n = net.forward(s, tracking=True)
        
            if index == 0:
                test_hidden = m
                test_output = n
            else:
                test_hidden = torch.cat((test_hidden, m), 1)
                test_output = torch.cat((test_output, n), 1)
        
            # Calculate the accuracy
            prediction = torch.argmax(s[0], dim=1)
            test_number += len(targets)
            correct_number += (prediction == targets).sum().item()

            #plot_decision_boundary(net, X, y)
        accuracy = float(correct_number/test_number)
        print('The accuracy is:', accuracy)
        trial.report(rep, accuracy)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        accuracy_list.append(float(correct_number/test_number))
        #plot_accuracy(accuracy_list)
        #plt.figure()
        #plot_decision_boundary(net, X, y)
    # Draw the evolution figure for the neurons
#    plot_decision_boundary(net, X, y)
# Plot the weights only for the selected epochs
    # for epoch, weights in epoch_weight_snapshots.items():
    #     plt.figure()
    #     plot_all_layers_weight_evolution_line([weights])  # Adjust your plotting function to work for a single epoch
    #     plt.title(f"Weight Evolution at Epoch {epoch}")
    #     plt.show()
    plot_accuracy(accuracy_list, act_fn_name)
    return accuracy_list[-1]
    
      
    
    #plot_evolution(train_output)


if __name__ == "__main__":
    # Ensure the directory exists
    directory = '/home/filip/optuna'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the path to the SQLite database file
    storage_url = "sqlite:///" + os.path.join(directory, "optuna_studies.db")
    study_name = "eq_prop_mnist_act"

    try:
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_url, load_if_exists=True)
        study.optimize(objective, n_trials=1000, timeout=72000)

        # Retrieve complete trials
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit()  # Exit if there's an error during study creation or optimization

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))