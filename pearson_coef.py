import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import pearsonr

plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams.update(
    {
        "ytick.direction": "in",
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "axes.grid": True,
        "xtick.direction": "in",
        "lines.linewidth": 2.0,
        "font.size": 13,
        "svg.fonttype": "none",
    }
)


def characterize(path: str, *, as_conductivity=False, plot=False):
    path = pathlib.Path(path)
    name = path.stem

    print(f"--- Statistics for `{name}` ---")

    with h5py.File(path, "r") as f:
        all_devices = np.asarray(list(f.values()))

    non_formed_devices = np.nonzero(all_devices[:, 0] > 30000)[0]

    select_formed_devices = np.ones(len(all_devices), dtype=np.bool_)
    select_formed_devices[non_formed_devices] = False
    devices = all_devices[select_formed_devices, :]

    pulses_count = devices.shape[1]

    if as_conductivity:
        devices = 1.0 / devices
        scale = 1e6
        unit = "µS"
    else:
        scale = 1e-3
        unit = "kΩ"

    pulses_mean = np.mean(devices, axis=0)
    pulses_median = np.median(devices, axis=0)
    pulses_std = np.std(devices, axis=0)

    pearson_coefs = [
        pearsonr(np.arange(pulses_count), device).correlation for device in devices
    ]

    print("- Total number of devices:", len(all_devices))
    print(
        "- Number of Non Formed Devices:",
        len(non_formed_devices),
        f"= {len(non_formed_devices)/len(all_devices)*100:.1f}%",
    )
    print("- Number of pulses per device:", pulses_count)
    print(
        f"- Starting value: Mean={pulses_mean[0]*scale:.1f}{unit}; Median={pulses_median[0]*scale:.1f}{unit}; Std={pulses_std[0]*scale:.1f}{unit}"
    )
    print(
        f"- End value: Mean={pulses_mean[-1]*scale:.1f}{unit}; Median={pulses_median[-1]*scale:.1f}{unit}"
    )
    print(
        f"- Value range: Mean={np.mean(devices[:, -1] - devices[:, 0])*scale:.1f}{unit}; Median={(pulses_median[-1]-pulses_median[0])*scale:.1f}{unit}"
    )
    print(
        f"- Pearson coef.: Mean={np.mean(pearson_coefs):.5f}; Std={np.std(pearson_coefs):.5f}"
    )
        

    if plot:
        fig, ax = plt.subplots(dpi=300)
        ax.plot(pulses_median * scale, label="Median")
        ax.plot(pulses_mean * scale, label="Mean", c="C2")
        ax.fill_between(
            np.arange(pulses_count),
            np.quantile(devices, 0.75, axis=0) * scale,
            np.quantile(devices, 0.25, axis=0) * scale,
            alpha=0.5,
            color="C1",
            label="Interquartile Range",
        )
        ax.set_title(
            f'{" ".join(map(str.capitalize, name.split("_")))} ({len(devices)} devices)'
        )
        ax.legend()
        ax.set_xlim(0, pulses_count)
        ax.set_ylabel(
            r"Resistance (k$\Omega$)" if not as_conductivity else r"Conductivity (µS)"
        )
        ax.set_xlabel("Number of pulses")
        plt.savefig(
            path.with_name(
                path.stem + "_stats" + ("_cond" if as_conductivity else "") + ".pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    return pearson_coefs, devices



def ideal_mem_model(path: str, fname ="ideal_model", as_conductivity=False, nb_sample=False):
    path = pathlib.Path(path)
    name = path.stem

    print(f"--- Statistics for `{name}` ---")

    with h5py.File(path, "r") as f:
        all_devices = np.asarray(list(f.values()))

    non_formed_devices = np.nonzero(all_devices[:, 0] > 30000)[0]

    select_formed_devices = np.ones(len(all_devices), dtype=np.bool_)
    select_formed_devices[non_formed_devices] = False
    devices = all_devices[select_formed_devices, :]

    pulses_count = devices.shape[1]

    if as_conductivity:
        devices = 1.0 / devices
        scale = 1e6
        unit = "µS"
    else:
        scale = 1e-3
        unit = "kΩ"

    pulses_mean = np.mean(devices, axis=0)
    pulses_median = np.median(devices, axis=0)
    pulses_std = np.std(devices, axis=0)

    file = h5py.File(f'{fname}', 'a')

    for idx in range (nb_sample):
        rand = np.random.rand(1)
        low_quantile_l = np.quantile(devices[:,0], 0.25, axis=0)
        high_quantile_l = np.quantile(devices[:,0], 0.75, axis=0)
        low_quantile_h = np.quantile(devices[:,-1], 0.25, axis=0)
        high_quantile_h = np.quantile(devices[:,-1], 0.75, axis=0)
        low = low_quantile_l + rand * (high_quantile_l - low_quantile_l)
        high = low_quantile_h + rand * (high_quantile_h - low_quantile_h)
        memristor_charcteristic = np.linspace(low[0], high[0], pulses_count, dtype=float)
        file.create_dataset(f'R{idx}', data=memristor_charcteristic)

    file.close()


def organize_pool(devices, pearson_coefs: list, thrs_min: float, thrs_max: float, range_min: int, fname: str):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with h5py.File(fname, 'a') as file:
        for idx, dev in enumerate(devices):
            if thrs_min <= pearson_coefs[idx] <= thrs_max and (dev.max() - dev.min()) >= range_min:
                file.create_dataset(f'R{idx}', data=dev)
                # Optionally plot each device's data
                # plt.plot(dev)
                # plt.title(f"Memristor resistance with Pearson coef of {pearson_coefs[idx]:.2f}")
                # plt.xlabel("Pulse count")
                # plt.ylabel(r"Resistance (k$\Omega$)")
                # plt.show()

def pearson_distribution(pearson_coefs):
    q1 = np.percentile(pearson_coefs, 25)
    q2 = np.percentile(pearson_coefs, 50)  # Median
    q3 = np.percentile(pearson_coefs, 75)
    print("Distribution and Quartiles:")
    print(f"First Quartile (Q1): {q1}")
    print(f"Median (Q2): {q2}")
    print(f"Third Quartile (Q3): {q3}")

    plt.hist(pearson_coefs, bins='auto', edgecolor='black')
    plt.xlabel('Pearson coeff.')
    plt.ylabel('Quantity of memristors')
    plt.title('Distribution of memristor Pearson coefficient')
    plt.show()

# Example use of the functions
original_file_path = "/home/filip/reram_data/march_slope_x3_5k.hdf5"
pearson_coefs, devices = characterize(original_file_path, plot=True)

# pearson_distribution(pearson_coefs)

pers_thrs_min = 0.86
pers_thrs_max = 1.0
range_min = 5000

file_str = f"WeakResets/march_slope_x3_5k_{range_min}_{pers_thrs_min}_to_{pers_thrs_max}.hdf5"
organize_pool(devices, pearson_coefs, pers_thrs_min, pers_thrs_max, range_min, file_str)