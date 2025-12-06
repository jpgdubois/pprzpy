import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background
from pprzpy.tools.signal_analysis import calc_signal_spectral_density

def plot_imu_gyro(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    """
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing raw GYRO messages.
    schema : Dict[str, pl.Schema]
        Schema dictionary for deserializing messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """

    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(2, 1)

    gyro_df = deserialize_payload(df, "IMU_GYRO_SCALED", schema)
    gyro_df = gyro_df.filter(pl.col("msg_timestamp") > 10)
    gyro_df = gyro_df.filter(pl.col("msg_timestamp") < 80)

    gyro_time = gyro_df["msg_timestamp"].to_numpy()
    gyro_x = gyro_df["gp"].to_numpy() * 0.0139882 * np.pi / 180
    gyro_y = gyro_df["gq"].to_numpy() * 0.0139882 * np.pi / 180
    gyro_z = gyro_df["gr"].to_numpy() * 0.0139882 * np.pi / 180

    gyro_x_freqs, gyro_x_psd = calc_signal_spectral_density(gyro_time, gyro_x)
    gyro_y_freqs, gyro_y_psd = calc_signal_spectral_density(gyro_time, gyro_y)
    gyro_z_freqs, gyro_z_psd = calc_signal_spectral_density(gyro_time, gyro_z)

    plot_signals(np.vstack([gyro_x, gyro_y, gyro_z]).T,
                 gyro_time,
                 signal_names=["Gyro X", "Gyro Y", "Gyro Z"],
                 colors=["blue", "orange", "green"],
                 ax=axs[0])
    axs[0].set_ylabel(r"Gyro $\frac{\textrm{rad}}{\textrm{s}}$")
    axs[0].legend(loc="lower left")

    plot_signals(np.vstack([gyro_x_psd, gyro_y_psd, gyro_z_psd]).T,
                 gyro_x_freqs,
                 signal_names=["Gyro X", "Gyro Y", "Gyro Z"],
                 colors=["blue", "orange", "green"],
                 ax=axs[1])
    axs[1].set_ylabel(r"Gyro $\frac{\textrm{rad}}{\textrm{s}}$")
    axs[1].legend(loc="lower left")
def plot_imu_gyro_derivative(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    """
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing raw GYRO messages.
    schema : Dict[str, pl.Schema]
        Schema dictionary for deserializing messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """

    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(2, 1)

    gyro_df = deserialize_payload(df, "IMU_GYRO_SCALED", schema)
    gyro_df = gyro_df.filter(pl.col("msg_timestamp") > 50)

    gyro_time = gyro_df["msg_timestamp"].to_numpy()
    gyro_x = gyro_df["gp"].to_numpy() * 0.0139882 * np.pi / 180
    gyro_y = gyro_df["gq"].to_numpy() * 0.0139882 * np.pi / 180
    gyro_z = gyro_df["gr"].to_numpy() * 0.0139882 * np.pi / 180

    # backward Euler (backward difference) derivative
    n = gyro_time.size
    if n <= 1:
        gyro_x = np.zeros_like(gyro_x)
        gyro_y = np.zeros_like(gyro_y)
        gyro_z = np.zeros_like(gyro_z)
    else:
        dt = np.diff(gyro_time)
        # avoid division by zero
        dt_safe = np.where(dt == 0, 1e-12, dt)

        gyro_x_dot = np.empty_like(gyro_x)
        gyro_y_dot = np.empty_like(gyro_y)
        gyro_z_dot = np.empty_like(gyro_z)

        # first sample: use forward difference (fallback)
        gyro_x_dot[0] = (gyro_x[1] - gyro_x[0]) / (gyro_time[1] - gyro_time[0]) if n > 1 else 0.0
        gyro_y_dot[0] = (gyro_y[1] - gyro_y[0]) / (gyro_time[1] - gyro_time[0]) if n > 1 else 0.0
        gyro_z_dot[0] = (gyro_z[1] - gyro_z[0]) / (gyro_time[1] - gyro_time[0]) if n > 1 else 0.0

        # backward Euler for remaining samples
        gyro_x_dot[1:] = np.diff(gyro_x) / dt_safe
        gyro_y_dot[1:] = np.diff(gyro_y) / dt_safe
        gyro_z_dot[1:] = np.diff(gyro_z) / dt_safe

        # replace signals with their derivatives for plotting
        gyro_x, gyro_y, gyro_z = gyro_x_dot, gyro_y_dot, gyro_z_dot

    gyro_x_freqs, gyro_x_psd = calc_signal_spectral_density(gyro_time, gyro_x_dot)
    gyro_y_freqs, gyro_y_psd = calc_signal_spectral_density(gyro_time, gyro_y_dot)
    gyro_z_freqs, gyro_z_psd = calc_signal_spectral_density(gyro_time, gyro_z_dot)

    plot_signals(np.vstack([gyro_x_dot, gyro_y_dot, gyro_z_dot]).T,
                 gyro_time,
                 signal_names=["Gyro X", "Gyro Y", "Gyro Z"],
                 colors=["blue", "orange", "green"],
                 ax=axs[0])
    axs[0].set_ylabel(r"Gyro $\frac{\textrm{rad}}{\textrm{s}}$")
    axs[0].legend(loc="lower left")

    plot_signals(np.vstack([gyro_x_psd, gyro_y_psd, gyro_z_psd]).T,
                 gyro_x_freqs,
                 signal_names=["Gyro X", "Gyro Y", "Gyro Z"],
                 colors=["blue", "orange", "green"],
                 ax=axs[1])
    axs[1].set_ylabel(r"Gyro $\frac{\textrm{rad}}{\textrm{s}}$")
    axs[1].legend(loc="lower left")