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
import sys

def one_hot_encode(labels, num_classes):
    # Create a tensor of zeros with shape (len(labels), num_classes)
    target = torch.zeros((len(labels), num_classes))
    
    # Use scatter_ to assign 1s in the correct class positions
    target.scatter_(1, labels.unsqueeze(1), 1)
    return target




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

        # Initialize DevicePool
        self.device_pool = DevicePool("/home/filip/reram_data/march_slope_x3_5k.hdf5")
        # Handle loss-specific configurations
        if loss == 'MSE':
            self.softmax_output = False
        elif loss == 'Cross-entropy':
            self.softmax_output = True

        # Initialize weight layers
        W = nn.ModuleList(None)
        self.Wn = []  # Initialize Wn as an empty list
        self.Wp = []  # Initialize Wp as an empty list
        self.devs_int = []
        
        
        #Initialize the device lists for the weights
        self.devs_int_p = []
        self.devs_int_n = []
        
        
        
        
        

        
        
        
        #Initialize the bias vectors
        self.bn = []  # Initialize Wn as an empty list
        self.bp = []  # Initialize Wp as an empty list
        
        #Initialize the device lists for the bias
        self.devs_int_pb = []
        self.devs_int_nb = []
        
        
        
        
        
        
        
        for i in range(len(fcLayers) - 1):
            

            # Get devs_int for this layer from the device pool
            layer_shape = (fcLayers[i + 1], fcLayers[i])  # Shape of weight matrix
            W.extend([nn.Linear(*layer_shape, bias=True)])
            
            self.Wn.append(np.zeros(((fcLayers[i], fcLayers[i + 1])), dtype=int))
            self.Wp.append(np.zeros(((fcLayers[i], fcLayers[i + 1])), dtype=int))
            
            self.bn.append(np.zeros((fcLayers[i],), dtype=int))
            self.bp.append(np.zeros((fcLayers[i],), dtype=int))


            devs_w = self.device_pool.request_couple((fcLayers[i], fcLayers[i + 1]))
            devs_b = self.device_pool.request_couple((fcLayers[i],))
            # Split into positive and negative matrices
            devs_pw, devs_nw = devs_w[0], devs_w[1]
            devs_pb, devs_nb = devs_b[0], devs_b[1]

            # Append to respective lists
            self.devs_int_p.append(devs_pw)
            self.devs_int_n.append(devs_nw)

            self.devs_int_pb.append(devs_pb)
            self.devs_int_nb.append(devs_nb)



        self.W = W





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


    def compute_gradients_ep(self, s, seq, weight_threshold = 0, target=None, noise_factor = None, baseline_distribution=None):
        """
        Compute EQ gradient to update the synaptic weight, with added noise including a random baseline.
        
        Parameters:
        - s: Current states.
        - seq: Sequential states.
        - target: Target labels.
        - noise_factor: Magnitude of the random noise.
        - baseline_distribution: A function that generates random values for the noise baseline (e.g., torch.randn).
        """
        batch_size = s[0].size(0)
        coef = 1 / (self.beta * batch_size)
    
        gradW, gradBias = [], []
        gradWp, gradWn = [], []
        gradbp, gradbn = [], [] 
        with torch.no_grad():
            # Calculate the gradients of the output layer
            if self.softmax_output:
                gradW.append(
                    -(0.5 / batch_size) * (torch.mm(torch.transpose((s[0] - target), 0, 1), self.act_fn.rho(s[1])) +
                                           torch.mm(torch.transpose((seq[0] - target), 0, 1),
                                                    self.act_fn.rho(seq[1]))))
                gradBias.append(-(0.5 / batch_size) * (s[0] + seq[0] - 2 * target).sum(0))
            else:
                gradW.append(coef * (torch.mm(torch.transpose(self.act_fn.rho(s[0]), 0, 1), self.act_fn.rho(s[1]))
                                     - torch.mm(torch.transpose(self.act_fn.rho(seq[0]), 0, 1),
                                                self.act_fn.rho(seq[1]))))
                gradBias.append(coef * (self.act_fn.rho(s[0]) - self.act_fn.rho(seq[0])).sum(0))
    
            # Calculate the gradients of the other layers
            for layer in range(1, len(s) - 1):
                gradW.append(coef * (torch.mm(torch.transpose(self.act_fn.rho(s[layer]), 0, 1), self.act_fn.rho(s[layer+1]))
                                     - torch.mm(torch.transpose(self.act_fn.rho(seq[layer]), 0, 1),
                                                self.act_fn.rho(seq[layer+1]))))
                gradBias.append(coef * (self.act_fn.rho(s[layer]) - self.act_fn.rho(seq[layer])).sum(0))
    
            
        weight_threshold = 0
        bias_threshold = 0
        
        for matrix in gradW:
            gradWp.append(torch.where(matrix > weight_threshold, 1, 0))  # Positive threshold
            gradWn.append(torch.where(matrix < -weight_threshold, 1, 0))  # Negative threshold
        
        for vector in gradBias:
            gradbp.append(torch.where(vector > bias_threshold, 1, 0))  # Positive threshold
            gradbn.append(torch.where(vector < -bias_threshold, 1, 0))  # Negative threshold
    
        # Add to the weight pulse matrix
        for (i, matrix) in enumerate(self.Wp):
            wp_array = wp_array = gradWp[i].cpu().numpy() 
            self.Wp[i] += wp_array
    

        for (i, matrix) in enumerate(self.Wn):
            wn_array = gradWn[i].cpu().numpy() 
            self.Wn[i] += wn_array


        # Add to the bias pulse matrix
        for (i, vector) in enumerate(self.bp):
            bp_array = gradbp[i].cpu().numpy() 
            self.bp[i] += bp_array
    

        for (i, vector) in enumerate(self.bn):
            bn_array = gradbn[i].cpu().numpy() 
            self.bn[i] += bn_array







    
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



    def update_W(self, gain = 5e3):
        
        for i, matrix in enumerate(self.W):
        # Compute new weights and biases using eval_weight
            devs = self.device_pool.devices
            devs_p = self.devs_int_p[i]
            devs_n = self.devs_int_n[i]
            pulses_p = self.Wp[i]
            pulses_n = self.Wn[i]
            new_weights = eval_weight(devs, devs_p, devs_n, pulses_p, pulses_n, gain)
            
            
            devs_b1_p = self.devs_int_pb[i]
            devs_b1_n = self.devs_int_nb[i]
            pulses_b1_p = self.bp[i]
            pulses_b1_n = self.bn[i]            
            
            
            
            
            new_bias = eval_weight(devs, devs_b1_p, devs_b1_n, pulses_b1_p, pulses_b1_n, gain),
    
        # Assign computed weights and biases to the layer
            if isinstance(matrix, nn.Linear):
                matrix.weight.data = torch.tensor(new_weights, dtype=matrix.weight.data.dtype)
                if matrix.bias is not None:
                    new_bias = np.array(new_bias)  # Convert list of NumPy arrays to a single NumPy array
                    matrix.bias.data = torch.tensor(new_bias, dtype=matrix.bias.data.dtype)
                    
                           
            
            
            
            #NEEDS TO BE FIXED
def init_zero_pulse(net, devs, devs_p, devs_n, pulses_p, pulses_n, gain):
    """
    Initializes the weights and biases of W layers using eval_weight.

    Args:
        net (mlp_eqprop): The network object containing W layers.
        devs (np.ndarray): Array of device properties for eval_weight.
        devs_p (np.ndarray): Positive device indices for eval_weight.
        devs_n (np.ndarray): Negative device indices for eval_weight.
        pulses_p (np.ndarray): Positive pulse values for eval_weight.
        pulses_n (np.ndarray): Negative pulse values for eval_weight.
        gain (float): Gain factor for eval_weight.
    """
    for i, matrix in enumerate(net.W):
        # Compute new weights and biases using eval_weight
        new_weights = eval_weight(devs, devs_p, devs_n, pulses_p, pulses_n, gain)
        
        # Assign computed weights and biases to the layer
        if isinstance(matrix, nn.Linear):
            matrix.weight.data = torch.tensor(new_weights, dtype=matrix.weight.data.dtype)
            if matrix.bias is not None:
                matrix.bias.data = torch.tensor(new_weights, dtype=matrix.bias.data.dtype)




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

def train_network(net, train_loader, test_loader, threshold, alt_layer, rep1, rep2, optimizer, epochs):
    
    sim_key = 'mnist_simulation'

    params = simulation_params[sim_key]
    
    accuracy_list = []
    epochs_to_plot = [0, 5, 10, 15, 20]
    
    # Initial learning rates
    lr_first = params['lr_first']
    lr_second = params['lr_second']
    lr = [lr_first, lr_second]
    
    #Initialize optimizer
    net_params, optimizer = defineOptimizer(net, lr, 'SGD')

    
    
    # Set initial epoch
    epoch = 0
    plot_all_linear_layer_weights_in_modulelist(net, epoch, epochs_to_plot)
    weight_threshold = 0
    for rep in range(epochs):
        
        
        if rep > rep1:
            weight_threshold = threshold
        
        
        
        if rep > rep2:
            weight_threshold = 1.5*threshold
        
        
        if (rep // alt_layer) % 2 == 0:  # Train the second layer
            for param_group in optimizer.param_groups[0:1]:  # First group (lr_first)
                param_group['lr'] = 0
            for param_group in optimizer.param_groups[1:2]:  # Second group (lr_second)
                param_group['lr'] = lr_second
        else:  # Train the first layer
            for param_group in optimizer.param_groups[1:2]:  # Second group (lr_second)
                param_group['lr'] = 0
            for param_group in optimizer.param_groups[0:1]:  # First group (lr_first)
                param_group['lr'] = lr_first
                
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)
            targets = one_hot_encode(targets, 10)
            s = net.init_state(data)
            s, m, n = net.forward(s, tracking=True)
            seq = s.copy()
            s, m, n = net.forward(s, beta=net.beta, target=targets, tracking=True)
            net.compute_gradients_ep(s, seq, weight_threshold, targets)
            optimizer.step()
            net.update_W()
    
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
        plot_all_linear_layer_weights_in_modulelist(net, rep +1, epochs_to_plot)
        print(f'Epoch {rep+1}: The accuracy is {accuracy:.2%}')
    
    return accuracy_list

def objective(trial):
    
    try:
        # Initialize and load your network here (net)
        sim_key = 'mnist_simulation'
        params = simulation_params[sim_key]
        
        
        threshold = trial.suggest_float('threshold', 1e-6, 1e-4)
        alt_layer = trial.suggest_int('alt_layer',1, 16)
        
        rep1 = trial.suggest_int('rep1',28, 40)
        rep2 = trial.suggest_int('rep1',40, 60)
        
        
        beta = trial.suggest_float('beta', 0.4, 1)
        batch_size = 1024
        # Load and preprocess data
        train_dataset, test_dataset = load_and_preprocess_data(
            dataset_name=params['dataset_name'],
            test_size=params['test_size'],
            random_state=params['random_state']
        )
        train_loader, test_loader = create_data_loaders(
            train_dataset,
            test_dataset,
            batch_size=batch_size
        )
        
        net = mlp_eqprop(
            fcLayers=params['fcLayers'],
            dt=params['dt'],
            T=params['T'],
            Kmax=params['Kmax'],
            beta=beta,
            loss=params['loss'],
            act_fn=params['activation_function'],
            data=params['data']
            )
        
        
        
    
        #Initialize W
        net.update_W()
        
        
        accuracy_list = train_network(net, train_loader, test_loader,  threshold, alt_layer, rep1, rep2, optimizer = None, epochs = params['epochs'])   
        print(batch_size)
        plot_accuracy(accuracy_list)
        
        return max(accuracy_list)
    
    
    except IndexError as e:
        # Handle the case where an IndexError occurs
        print(f"Encountered an IndexError: {e} - Skipping this trial.")
        return None  # Optuna will treat this trial as failed and continue
    
    
    
def main():
    directory = '/home/filip/optuna'
    if not os.path.exists(directory):
        os.makedirs(directory)

    storage_url = "sqlite:///" + os.path.join(directory, "optuna_studies.db")
    study_name = "eq_prop_memnist_all_mem"

    try:
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_url, load_if_exists=True)
        study.optimize(objective, n_trials=1000, timeout=72000)

        # Retrieve complete trials
        complete_trials = [trial for trial in study.get_trials() if trial.state == TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    except Exception as e:
        print(f"An error occurred: {e}")
        # Instead of exiting, handle or log the error and potentially restart or adjust optimization
        # Here you could initiate a recovery or backup plan

if __name__ == "__main__":
    main()