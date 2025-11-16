import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background

def plot_attitude(stab_attitude_df: pl.DataFrame, rotorcraft_status_df: pl.DataFrame, ax: plt.Axes):
    """
    Plot stabilized attitude Euler angles with autopilot mode background and status lines.

    Parameters
    ----------
    stab_attitude_df : pl.DataFrame
        DataFrame containing deserialized STAB_ATTITUDE messages.
    rotorcraft_status_df : pl.DataFrame
        DataFrame containing deserialized ROTORCRAFT_STATUS messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """
    rotorcraft_status_time = rotorcraft_status_df["timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    stab_attitude_time = stab_attitude_df["timestamp"].to_numpy()
    att_des_quat = stab_attitude_df["att_des"].to_numpy()
    att_ref_quat = stab_attitude_df["att_ref"].to_numpy()
    att_quat = stab_attitude_df["att"].to_numpy()

    def _sanitize_quats(q_array):
        q = np.asarray(q_array)
        if q.dtype == object:
            q = np.vstack(q)
        if q.ndim == 1 and q.size == 4:
            q = q.reshape(1, 4)
        if q.ndim != 2 or q.shape[1] != 4:
            raise ValueError("Quaternions must have shape (N,4)")
        q = np.array(q, dtype=float, copy=True)
        norms = np.linalg.norm(q, axis=1)
        small = norms < 1e-3
        if np.any(small):
            q[small] = np.array([1.0, 0.0, 0.0, 0.0])
            norms = np.linalg.norm(q, axis=1)
        q = q / norms[:, None]
        return q

    att_des_quat = _sanitize_quats(att_des_quat)
    att_ref_quat = _sanitize_quats(att_ref_quat)
    att_quat = _sanitize_quats(att_quat)

    att_des_euler = Rotation.from_quat(att_des_quat).as_euler("zyx", degrees=False)
    att_ref_euler = Rotation.from_quat(att_ref_quat).as_euler("zyx", degrees=False)
    att_euler = Rotation.from_quat(att_quat).as_euler("zyx", degrees=False)

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=ax, color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=ax,
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    # plot_signals(att_des_euler, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"],
    #              signal_names=["roll_des", "pitch_des", "yaw_des"], linestyle="dotted", discrete=True)
    plot_signals(att_ref_euler, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"],
                 signal_names=["roll_ref", "pitch_ref", "yaw_ref"], linestyle="dashed", discrete=True)
    plot_signals(att_euler, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"], signal_names=["roll", "pitch", "yaw"])
    ax.set_ylabel(r"Attitude $\textrm{rad}$")
    ax.legend(loc="lower left")


def plot_rates(stab_attitude_df: pl.DataFrame, rotorcraft_status_df: pl.DataFrame, ax: plt.Axes):
    """
    Plot angular rate references and measurements with autopilot mode background and status lines.

    Parameters
    ----------
    stab_attitude_df : pl.DataFrame
        DataFrame containing deserialized STAB_ATTITUDE messages.
    rotorcraft_status_df : pl.DataFrame
        DataFrame containing deserialized ROTORCRAFT_STATUS messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """
    rotorcraft_status_time = rotorcraft_status_df["timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    stab_attitude_time = stab_attitude_df["timestamp"].to_numpy()
    rate_ref = stab_attitude_df["angular_rate_ref"].to_numpy()
    rate = stab_attitude_df["angular_rate"].to_numpy()

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=ax, color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=ax,
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    plot_signals(rate_ref, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"],
                 signal_names=["p_ref", "q_ref", "r_ref"], linestyle="dashed", discrete=True)
    plot_signals(rate, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"], signal_names=["p", "q", "r"])
    ax.set_ylabel(r"Angular rates $\frac{\textrm{rad}}{\textrm{s}}$")
    ax.legend(loc="lower left")


def plot_accelerations(stab_attitude_df: pl.DataFrame, rotorcraft_status_df: pl.DataFrame, ax: plt.Axes):
    """
    Plot angular acceleration references and measurements with autopilot mode background and status lines.

    Parameters
    ----------
    stab_attitude_df : pl.DataFrame
        DataFrame containing deserialized STAB_ATTITUDE messages.
    rotorcraft_status_df : pl.DataFrame
        DataFrame containing deserialized ROTORCRAFT_STATUS messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """
    rotorcraft_status_time = rotorcraft_status_df["timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    stab_attitude_time = stab_attitude_df["timestamp"].to_numpy()
    accel_ref = stab_attitude_df["angular_accel_ref"].to_numpy()
    accel = stab_attitude_df["angular_accel"].to_numpy()

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=ax, color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=ax,
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    plot_signals(accel_ref, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"],
                 signal_names=["p_dot_ref", "q_dot_ref", "r_dot_ref"], linestyle="dashed", discrete=True)
    plot_signals(accel, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"], signal_names=["p_dot", "q_dot", "r_dot"])
    ax.set_ylabel(r"Angular accelerations $\frac{\textrm{rad}}{\textrm{s}^2}$")
    ax.legend(loc="lower left")


def plot_stab_attitude(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    """
    Wrapper function to deserialize data and plot attitude, rates, and accelerations on 3 subplots.

    Parameters
    ----------
    df : pl.DataFrame
        Raw telemetry data.
    schema : Dict[str, pl.Schema]
        Dictionary mapping message types to Polars schemas.
    """
    plt.rcParams['text.usetex'] = True

    stab_attitude_df = deserialize_payload(df, "STAB_ATTITUDE", schema)
    rotorcraft_status_df = deserialize_payload(df, "ROTORCRAFT_STATUS", schema)

    fig, axs = plt.subplots(3, 1, sharex=True)

    plot_attitude(stab_attitude_df, rotorcraft_status_df, axs[0])
    plot_rates(stab_attitude_df, rotorcraft_status_df, axs[1])
    plot_accelerations(stab_attitude_df, rotorcraft_status_df, axs[2])
