import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background

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
    if not rotorcraft_status_df.is_empty():
        fill_signal_condition_background(
            rotorcraft_status_df["ap_mode"].to_numpy(),
            rotorcraft_status_df["msg_timestamp"].to_numpy(),
            conditions={
                Condition(0, "Kill", "red"),
                Condition(3, "Rate", "yellow"),
                Condition(4, "Atti", "green")
            },
            annotate=True, alpha=0.15,
            ax=ax,
        )
        draw_signal_status_lines(
            rotorcraft_status_df["ap_in_flight"].to_numpy(), 
            rotorcraft_status_df["msg_timestamp"].to_numpy(), 
            label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
            linewidth=1.0, linestyle="dashed",
            ax=ax,
        )

    stab_attitude_time = stab_attitude_df["msg_timestamp"].to_numpy()
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

    # plot_signals(att_des_euler, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"],
    #              signal_names=["roll_des", "pitch_des", "yaw_des"], linestyle="dotted", discrete=True)
    # plot_signals(att_ref_euler, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"],
    #              signal_names=["roll_ref", "pitch_ref", "yaw_ref"], linestyle="dashed", discrete=True)
    # plot_signals(att_euler, stab_attitude_time, ax=ax, colors=["red", "green", "darkblue"], signal_names=["roll", "pitch", "yaw"])
    # ax.set_ylabel(r"Attitude $\textrm{rad}$")
    plot_signals(att_des_quat, stab_attitude_time, ax=ax, colors=["cyan", "magenta", "yellow", "black"],
                 signal_names=["q_w_des", "q_x_des", "q_y_des", "q_z_des"], linestyle="dotted", discrete=True)
    plot_signals(att_ref_quat, stab_attitude_time, ax=ax, colors=["cyan", "magenta", "yellow", "black"],
                 signal_names=["q_w_ref", "q_x_ref", "q_y_ref", "q_z_ref"], linestyle="dashed", discrete=True)
    plot_signals(att_quat, stab_attitude_time, ax=ax, colors=["cyan", "magenta", "yellow", "black"],
                 signal_names=["q_w", "q_x", "q_y", "q_z"], linestyle="solid")
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
    if not rotorcraft_status_df.is_empty():
        fill_signal_condition_background(
            rotorcraft_status_df["ap_mode"].to_numpy(),
            rotorcraft_status_df["msg_timestamp"].to_numpy(),
            conditions={
                Condition(0, "Kill", "red"),
                Condition(3, "Rate", "yellow"),
                Condition(4, "Atti", "green")
            },
            annotate=True, alpha=0.15,
            ax=ax,
        )
        draw_signal_status_lines(
            rotorcraft_status_df["ap_in_flight"].to_numpy(), 
            rotorcraft_status_df["msg_timestamp"].to_numpy(), 
            label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
            linewidth=1.0, linestyle="dashed",
            ax=ax,
        )

    stab_attitude_time = stab_attitude_df["msg_timestamp"].to_numpy()
    rate_ref = stab_attitude_df["angular_rate_ref"].to_numpy()
    rate = stab_attitude_df["angular_rate"].to_numpy()

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
    if not rotorcraft_status_df.is_empty():
        fill_signal_condition_background(
            rotorcraft_status_df["ap_mode"].to_numpy(),
            rotorcraft_status_df["msg_timestamp"].to_numpy(),
            conditions={
                Condition(0, "Kill", "red"),
                Condition(3, "Rate", "yellow"),
                Condition(4, "Atti", "green")
            },
            annotate=True, alpha=0.15,
            ax=ax,
        )
        draw_signal_status_lines(
            rotorcraft_status_df["ap_in_flight"].to_numpy(), 
            rotorcraft_status_df["msg_timestamp"].to_numpy(), 
            label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
            linewidth=1.0, linestyle="dashed",
            ax=ax,
        )

    stab_attitude_time = stab_attitude_df["msg_timestamp"].to_numpy()
    accel_ref = stab_attitude_df["angular_accel_ref"].to_numpy()
    accel = stab_attitude_df["angular_accel"].to_numpy()

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

    nu_df = deserialize_payload(df, "EFF_MAT_STAB", schema)
    actuators_df = deserialize_payload(df, "ACTUATORS_T4_IN", schema)


    # Get control effectiveness matrix
    ce_roll = nu_df["G1_roll"].to_numpy()
    ce_pitch = nu_df["G1_pitch"].to_numpy()
    ce_yaw = nu_df["G1_yaw"].to_numpy()
    ce_thrust = nu_df["G1_thrust"].to_numpy()

    ce_mat = np.stack([ce_roll, ce_pitch, ce_yaw, ce_thrust], axis=-1)

    ce_timestamp = nu_df["msg_timestamp"].to_numpy()

    plot_attitude(stab_attitude_df, rotorcraft_status_df, axs[0])
    plot_rates(stab_attitude_df, rotorcraft_status_df, axs[1])
    plot_accelerations(stab_attitude_df, rotorcraft_status_df, axs[2])
