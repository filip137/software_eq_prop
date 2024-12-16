
import numpy as np
import matplotlib.pyplot as plt
from DevicePool import *

def plot_first_10_devices_and_average(device_array):
    # Extract the first 10 devices (assuming "first 10" was intended)
    first_10_devices = device_array[:200, :]

    # Calculate the average across all values for each iteration among the first 10 devices
    average_values = np.mean(first_10_devices, axis=0)

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plot each of the first 10 device's readings over all iterations
    for device_values in first_10_devices:
        plt.plot(device_values, alpha=0.6)  # Adjusted alpha for clarity

    # Plot the average value at each iteration
    plt.plot(average_values, color='red', marker='x', linestyle='-', label='Average Value of First 10 Devices')

    plt.title('Device Values Over All Iterations for the First 10 Devices')
    plt.xlabel('Iteration Index')
    plt.ylabel('Device Values')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Assuming you have already initialized your DevicePool and loaded your device array
    pool = DevicePool("/home/filip/reram_data/march_slope_x3_5k.hdf5")
    device_values = pool.devices

    # Plot the first 50 values and their average
    plot_first_10_devices_and_average(device_values)
