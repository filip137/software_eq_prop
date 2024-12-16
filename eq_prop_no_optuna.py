import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from DevicePool import *
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
import optuna
from plots_eq_prop import *
from sklearn.preprocessing import StandardScaler
import os
from sklearn.datasets import make_moons
from data_processing import load_and_preprocess_data, create_data_loaders
from config import simulation_params
from plots_eq_prop import *

def one_hot_encode(labels, num_classes):
    # Create a tensor of zeros with shape (len(labels), num_classes)
    target = torch.zeros((len(labels), num_classes))
    
    # Use scatter_ to assign 1s in the correct class positions
    target.scatter_(1, labels.unsqueeze(1), 1)
    return target



    
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


class mlp_eqprop(nn.Module):
    def __init__(self, fcLayers, dt, T, Kmax, beta, loss, act_fn, data = None):
        """
        Initializes the mlp_eqprop neural network with DevicePool and pulse matrices.

        Args:
            fcLayers (list): Layer sizes for the neural network.
            dt (float): Time step for dynamics.
            T (int): Total number of time steps.
            Kmax (int): Maximum iterations.
            beta (float): Scaling factor for dynamics.
            loss (str): Loss function ('MSE' or 'Cross-entropy').
            act_fn (callable): Activation function.
            data (str or Path): Path to data for the DevicePool.
        """
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
            # weights = self.W[0].weight  # This will be of shape [10, 128] for out_features=10, in_features=128
            # biases = self.W[0].bias    # This will be of shape [10]
            # s1_activated = self.act_fn.rho(s[1]) 
            # linear_output = torch.mm(s1_activated, weights) 
            # s0_rhop = self.act_fn.rhop(s[0])
            # dsdt = [-s[0] + (s0_rhop * linear_output)]
            
            
            # print("Shape of input to W[0]:", s[1].shape)
            # print("Expected in_features for W[0]:", self.W[0].in_features)
            
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform_(m.weight)  # Xavier uniform
        # Or Xavier normal
        torch.nn.init.xavier_normal_(m.weight) 

def train_network(net, train_loader, test_loader, optimizer, epochs):
    accuracy_list = []
    epochs_to_plot = [0, 5, 10, 15, 20]
    epoch = 0
    plot_all_linear_layer_weights_in_modulelist(net, epoch, epochs_to_plot)
    for rep in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)
            targets = one_hot_encode(targets, 10)
            s = net.init_state(data)
            s, m, n = net.forward(s, tracking=True)
            seq = s.copy()
            s, m, n = net.forward(s, beta=net.beta, target=targets, tracking=True)
            net.compute_gradients_ep(s, seq, targets)
            net.compute_gradients_ep(s, seq, targets)
            optimizer.step()
    
        # Testing phase
        correct_number, test_number = 0, 0
        for data, targets in test_loader:
            data = data.view(data.size(0), -1)
            s = net.init_state(data)
            s, m, n = net.forward(s, tracking=True)
            prediction = torch.argmax(s[0], dim=1)
            correct_number += (prediction == targets).sum().item()
            test_number += len(targets)
    
        accuracy = float(correct_number) / test_number
        accuracy_list.append(accuracy)
        plot_all_linear_layer_weights_in_modulelist(net, rep, epochs_to_plot)
        print(f'Epoch {rep+1}: The accuracy is {accuracy:.2%}')
    
    return accuracy_list

def main():
    # Initialize and load your network here (net)
    sim_key = 'mnist_simulation'
    params = simulation_params[sim_key]
    # Load and preprocess data
    train_dataset, test_dataset = load_and_preprocess_data(
        dataset_name=params['dataset_name'],
        test_size=params['test_size'],
        random_state=params['random_state']
    )
    train_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=params['batch_size']
    )
    
    net = mlp_eqprop(
        fcLayers=params['fcLayers'],
        dt=params['dt'],
        T=params['T'],
        Kmax=params['Kmax'],
        beta=params['beta'],
        loss=params['loss'],
        act_fn=params['activation_function'],
        data=params['data']
        )
    
    lr = [params['lr_first'], params['lr_second']]
    net_params, optimizer = defineOptimizer(net, lr, 'SGD')
    net.apply(init_weights)
    
    
    accuracy_list = train_network(net, train_loader, test_loader, optimizer,epochs=300)   
    
    plot_accuracy(accuracy_list)

if __name__ == "__main__":
    main()
