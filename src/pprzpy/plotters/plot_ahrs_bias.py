import polars as pl
from typing import Dict, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background

def plot_ahrs_bias(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    ahrs_bias_df = deserialize_payload(df, "AHRS_BIAS", schema)

    timestamp = ahrs_bias_df["msg_timestamp"].to_numpy()
    accel_x = ahrs_bias_df["accel_x"].to_numpy()
    accel_y = ahrs_bias_df["accel_y"].to_numpy()
    accel_z = ahrs_bias_df["accel_z"].to_numpy()
    gyro_p = ahrs_bias_df["gyro_p"].to_numpy()
    gyro_q = ahrs_bias_df["gyro_q"].to_numpy()
    gyro_r = ahrs_bias_df["gyro_r"].to_numpy()
    mag_x = ahrs_bias_df["mag_x"].to_numpy()
    mag_y = ahrs_bias_df["mag_y"].to_numpy()
    mag_z = ahrs_bias_df["mag_z"].to_numpy()

    fix, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    accel = np.stack([accel_x, accel_y, accel_z], axis=1)
    gyro = np.stack([gyro_p, gyro_q, gyro_r], axis=1)
    mag = np.stack([mag_x, mag_y, mag_z], axis=1)

    plot_signals(accel, timestamp, colors=["red", "green", "blue"],
                 signal_names=["Accel X Bias", "Accel Y Bias", "Accel Z Bias"],
                 linestyle="solid",
                 discrete=False,
                 marker=".",
                 ax=axs[0])
    plot_signals(gyro, timestamp, colors=["red", "green", "blue"],
                 signal_names=["Gyro P Bias", "Gyro Q Bias", "Gyro R Bias"],
                 linestyle="solid",
                 discrete=False,
                 marker=".",
                 ax=axs[1])
    plot_signals(mag, timestamp, colors=["red", "green", "blue"],
                 signal_names=["Mag X Bias", "Mag Y Bias", "Mag Z Bias"],
                 linestyle="solid",
                 discrete=False,
                 marker=".",
                 ax=axs[2])
