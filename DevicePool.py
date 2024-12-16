import math
import logging
from pathlib import Path
import numpy as np
import h5py


class DevicePool:
    """
    A pool of devices loaded from an HDF5 file for managing device selections and reshuffling.
    """

    def __init__(self, filename, devices_count=None):
        """
        Initialize the DevicePool with devices loaded from an HDF5 file.

        Args:
            filename (str or Path): Path to the HDF5 file.
            devices_count (int, optional): Maximum number of devices to load. Defaults to None.
        """
        self._filename = Path(filename)

        with h5py.File(self._filename, "r") as f:
            self.devices = np.asarray(list(f.values()))

        if self.devices.ndim != 2:
            raise ValueError(
                f"Invalid devices shape, expected 2 dimensions, got {self.devices.ndim}"
            )

        if devices_count is not None:
            if devices_count > 0:
                if devices_count <= len(self.devices):
                    self.devices = self.devices[:devices_count]
                else:
                    logging.warning(
                        f"Initialized DevicePool with {devices_count} devices, but only {len(self.devices)} available in file '{filename}'"
                    )
            else:
                raise ValueError("devices_count must be a positive integer.")

        self.num_devices, self.num_measurements = self.devices.shape
        self._devices_order = np.arange(self.num_devices, dtype=np.uint32)
        np.random.shuffle(self._devices_order)
        self.pool_index = 0

    def request_couple(self, shape):
        """
        Request a couple of devices reshaped to the given shape.

        Args:
            shape (tuple): Shape of the requested device pairs.

        Returns:
            np.ndarray: Selected device indices reshaped to (2, *shape).
        """
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"Shape dimensions must be positive, got {shape}.")

        total_req_size = 2 * math.prod(shape)  # *2 for pos and neg
        devices_left_in_pool = len(self._devices_order) - self.pool_index

        if total_req_size > devices_left_in_pool:
            logging.warning(
                f"Out-of-devices: reshuffling the pool for request of shape {shape}"
            )

            sel_dev_chunks = [self._devices_order[self.pool_index:]]
            total_req_size -= devices_left_in_pool  # Empty the pool
            np.random.shuffle(self._devices_order)

            while total_req_size > len(self._devices_order):
                sel_dev_chunks.append(self._devices_order[:])
                total_req_size -= len(self._devices_order)
                np.random.shuffle(self._devices_order)

            self.pool_index = total_req_size
            sel_dev_chunks.append(self._devices_order[:self.pool_index])

            selected_devices = np.concatenate(sel_dev_chunks)
        else:
            selected_devices = self._devices_order[
                self.pool_index : self.pool_index + total_req_size
            ]
            self.pool_index += total_req_size

        return selected_devices.reshape((2, *shape))


def eval_weight(devs, devs_p, devs_n, pulses_p, pulses_n, gain):
    """
    Evaluate the weight for a given set of devices and pulses.

    Args:
        devs (np.ndarray): Array of devices.
        devs_p (np.ndarray): Positive device indices.
        devs_n (np.ndarray): Negative device indices.
        pulses_p (np.ndarray): Positive pulse values.
        pulses_n (np.ndarray): Negative pulse values.
        gain (float): Gain factor.

    Returns:
        np.ndarray: Evaluated weight.
    """
    return gain * (1.0 / devs[devs_n, pulses_n] - 1.0 / devs[devs_p, pulses_p])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pool = DevicePool("/home/filip/reram_data/march_slope_x3_5k.hdf5")
    device_values = pool.devices
    devs_W1_p, devs_W1_n = pool.request_couple((64, 128))
    devs_b1_p, devs_b1_n = pool.request_couple((3,))

    pulses_W1_p = np.zeros_like(devs_W1_p)
    pulses_W1_n = np.zeros_like(devs_W1_n)
    pulses_b1_p = np.zeros_like(devs_b1_p)
    pulses_b1_n = np.zeros_like(devs_b1_n)

    print(
        "W1",
        eval_weight(pool.devices, devs_W1_p, devs_W1_n, pulses_W1_p, pulses_W1_n, 5e3),
        "\nb1",
        eval_weight(pool.devices, devs_b1_p, devs_b1_n, pulses_b1_p, pulses_b1_n, 5e3),
    )
