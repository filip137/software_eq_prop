import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math




def plot_losses(losses):
    """
    Plots the progression of loss values.

    Parameters:
    losses (list of float): A list containing loss values recorded over training epochs or iterations.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))  # Set the figure size for the plot
    plt.plot(losses, marker='o', linestyle='-', color='blue')  # Plot the losses with markers and a line
    plt.title('Loss Progression Over Time')  # Set the title of the plot
    plt.xlabel('Epoch / Iteration')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.grid(True)  # Enable grid for easier readability
    plt.show()  # Display the plot

def plot_decision_boundary(model, X, y, grid_step=0.05, cmap=plt.cm.Paired):
    # Set the model to evaluation mode
    model.eval()

    # Generate a grid of points covering the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    
    # Convert the grid to a tensor and pass it through the model
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z  = []
    with torch.no_grad():
        for data_point in grid:
            data_point = data_point.reshape((1,2))
            s = model.init_state(data_point) #write the inputs
            s, m, n = model.forward(s, tracking=True)
            if s[0].size()[-1] > 1:
                result = s[0][0][1].float()
            else:
                result = s[0].float()
            
            if result > 0.5:
                prediction = 1
            else:
                prediction = 0 
            Z.append(prediction)
            
    Z = np.array(Z).reshape(xx.shape)
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()  

def plot_accuracy(accuracy_list):
    """
    Plots the accuracy list over epochs or iterations.
    
    Parameters:
    accuracy_list (list): A list of accuracy values to plot.
    
    Returns:
    None: Displays a plot of the accuracy values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy_list, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.title('Accuracy over Time')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_neuron_outputs(neuron_values):
    """
    Plots the values of output neurons over time or iterations.

    Parameters:
    neuron_values (list of torch.Tensor): A list where each element is a tensor
    containing the output values of neurons at a specific time or iteration.

    Returns:
    None
    """
    # Prepare the figure
    plt.figure(figsize=(10, 5))
    # Assuming each tensor in the list represents an iteration or a state
    for i, values in enumerate(neuron_values):
        plt.plot(values.numpy(), marker='o', linestyle='-', label=f'Iteration {i+1}')

    plt.title('Output Neuron Values Over Iterations')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Output Value')
    #plt.legend()
    plt.grid(True)
    plt.show()

def plot_neuron_statistics(neuron_values):
    """
    Plots the min, max, and average values of each neuron over all iterations.

    Parameters:
    neuron_values (list of torch.Tensor): A list where each element is a tensor
    containing the output values of neurons at a specific time or iteration.

    Returns:
    None
    """
    # Convert list of tensors to a single 2D tensor
    neuron_matrix = torch.stack(neuron_values)

    # Calculate the mean, min, and max across all iterations for each neuron
    means = torch.mean(neuron_matrix, dim=0)
    mins = torch.min(neuron_matrix, dim=0)[0]
    maxs = torch.max(neuron_matrix, dim=0)[0]

    # Prepare the figure
    plt.figure(figsize=(12, 6))

    # Plotting the average of the neuron outputs
    plt.plot(means.numpy(), marker='o', linestyle='-', label='Average Neuron Output', color='blue')
    
    # Plotting the min values
    plt.plot(mins.numpy(), marker='o', linestyle='--', label='Min Neuron Output', color='red')

    # Plotting the max values
    plt.plot(maxs.numpy(), marker='o', linestyle='--', label='Max Neuron Output', color='green')
    
    plt.title('Neuron Output Statistics Across All Iterations')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Output Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_neuron_stats(neuron_values):
#     """
#     Plots the average, minimum, and maximum neuron values over time or iterations.

#     Parameters:
#     neuron_values (list of torch.Tensor): A list where each element is a tensor
#     containing the output values of neurons at a specific time or iteration.

#     Returns:
#     None
#     """
#     # Convert list of tensors to a single 2D tensor
#     neuron_matrix = torch.stack(neuron_values)

#     # Calculate the mean, min, and max across each iteration
#     means = torch.mean(neuron_matrix, dim=1)
#     mins = torch.min(neuron_matrix, dim=1)[0]
#     maxs = torch.max(neuron_matrix, dim=1)[0]

#     # Prepare the figure
#     plt.figure(figsize=(12, 6))
    
#     # Plotting the mean of the neuron outputs
#     plt.plot(means.numpy(), label='Mean Neuron Output', color='blue', marker='o')
    
#     # Fill between the range of min and max neuron outputs
#     plt.fill_between(range(len(mins)), mins.numpy(), maxs.numpy(), color='gray', alpha=0.3, label='Range (Min-Max)')
    
#     plt.title('Neuron Output Statistics Over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Neuron Output Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def collect_weights(module_list):
    weight_snapshots = []
    for layer in module_list:
        weight_snapshots.append(layer.weight.data.clone())  # Clone the weights to keep a snapshot
    return weight_snapshots




def plot_weight_evolution(weight_history):
    num_layers = len(weight_history[0])
    num_iterations = len(weight_history)
    
    for layer_idx in range(num_layers):
        fig, ax = plt.subplots(figsize=(10, 8))
        # Create an array to hold all the weights for this layer
        weights = np.array([snapshot[layer_idx].numpy().flatten() for snapshot in weight_history])
        
        # Using imshow to visualize the evolution of weights
        cax = ax.imshow(weights, aspect='auto', cmap='viridis')
        ax.set_title(f'Evolution of weights in Layer {layer_idx}')
        ax.set_xlabel('Weight Index')
        ax.set_ylabel('Iteration')
        fig.colorbar(cax)
        plt.show()

def plot_all_layers_weight_evolution_line(weight_history):
    num_iterations = len(weight_history)
    num_layers = len(weight_history[0])
    
    # Iterate over each layer in the ModuleList
    for layer_index in range(num_layers):
        # Extract the weights for the specified layer across all iterations
        weights = np.array([snapshot[layer_index].numpy().flatten() for snapshot in weight_history])
        
        # Create a figure for each layer
        plt.figure(figsize=(12, 8))
        for weight_idx in range(weights.shape[1]):  # Iterate over each weight
            plt.plot(range(num_iterations), weights[:, weight_idx], label=f'Weight {weight_idx}')

        plt.title(f'Evolution of weights in Layer {layer_index}')
        plt.xlabel('Iteration')
        plt.ylabel('Weight Value')
        #plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.show()




def plot_evolution(x):
    # Determine the number of subplots needed based on the first dimension of x
    n_components = x.size(0)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(n_components, 1, figsize=(10, n_components * 3))
    
    # Iterate over each component of x
    for idx in range(n_components):
        # Check if axes is an array (when n_components is 1, it's not)
        if n_components == 1:
            ax = axes
        else:
            ax = axes[idx]
        
        # Plot the data on the respective subplot
        ax.plot(x[idx, :], label=f'Neuron {idx}')
        ax.set_ylabel('Neuron state')
        ax.set_xlabel('Time')
        #ax.legend(loc='upper right')
    
    # Adjust the layout so labels do not overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    

def plot_evolution_grid(x):
    plt.figure()
#    plt.clf()  # Clear the current figure
    n_components = x.size(0)
    rows = int(math.sqrt(n_components))
    cols = math.ceil(n_components / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    for idx in range(n_components):
        ax = axes[idx // cols, idx % cols] if n_components > 1 else axes.flat[0]
        ax.plot(x[idx, :], label=f'Neuron {idx}')
        ax.set_ylabel('Neuron state')
        ax.set_xlabel('Time')
        #ax.legend(loc='upper right')

    plt.tight_layout()
    plt.pause(0.01)  # A small pause to update the figure    
    

# Draw the weights 
def imshow_weight(w):
    fig,ax = plt.subplots(1, len(w))
    for i in range(len(w)):
        ax[i].imshow(w[i].weight.detach().numpy(), cmap=cm.coolwarm)
        ax[i].set_ylabel(f'Layer {i}')
    fig.suptitle('Weight values')
    plt.tight_layout()
    plt.subplots_adjust(top=1.2)
