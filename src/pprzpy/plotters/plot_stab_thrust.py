import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background

def plot_thrust(stab_thrust_df: pl.DataFrame, rotorcraft_status_df: pl.DataFrame, ax: plt.Axes):
    """
    Parameters
    ----------
    stab_thrust_df : pl.DataFrame
        DataFrame containing deserialized STAB_ATTITUDE messages.
    rotorcraft_status_df : pl.DataFrame
        DataFrame containing deserialized ROTORCRAFT_STATUS messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """
    rotorcraft_status_time = rotorcraft_status_df["timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    stab_thrust_time = stab_thrust_df["timestamp"].to_numpy()
    thrust_des = np.atleast_2d(stab_thrust_df["thrust_des"].to_numpy()).T
    thrust_ref = np.atleast_2d(stab_thrust_df["thrust_ref"].to_numpy()).T
    thrust = np.atleast_2d(stab_thrust_df["thrust_state"].to_numpy()).T

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=ax, color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=ax,
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    # plot_signals(thrust_des, stab_thrust_time, ax=ax, colors=["red"],
    #              signal_names=["thrust_des"], linestyle="dotted", discrete=True)
    plot_signals(thrust_ref, stab_thrust_time, ax=ax, colors=["green"],
                 signal_names=["thrust_ref"], linestyle="dashed", discrete=True)
    plot_signals(thrust, stab_thrust_time, ax=ax, colors=["darkblue"], signal_names=["thrust_state"])
    ax.set_ylabel(r"Specific thrust $\frac{\textrm{m}}{\textrm{s}^2}$")
    ax.legend(loc="lower left")

def plot_thrust_rate(stab_thrust_df: pl.DataFrame, rotorcraft_status_df: pl.DataFrame, ax: plt.Axes):
    """
    Parameters
    ----------
    stab_thrust_df : pl.DataFrame
        DataFrame containing deserialized STAB_ATTITUDE messages.
    rotorcraft_status_df : pl.DataFrame
        DataFrame containing deserialized ROTORCRAFT_STATUS messages.
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    """
    rotorcraft_status_time = rotorcraft_status_df["timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    stab_thrust_time = stab_thrust_df["timestamp"].to_numpy()
    thrust_d_ref = np.atleast_2d(stab_thrust_df["thrust_d_ref"].to_numpy()).T

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=ax, color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=ax,
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    plot_signals(thrust_d_ref, stab_thrust_time, ax=ax, colors=["green"],
                 signal_names=["thrust_d_ref"], linestyle="dashed", discrete=True)
    ax.set_ylabel(r"Specific thrust rate $\frac{\textrm{m}}{\textrm{s}^3}$")
    ax.legend(loc="lower left")


def plot_stab_thrust(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    """
    Parameters
    ----------
    df : pl.DataFrame
        Raw telemetry data.
    schema : Dict[str, pl.Schema]
        Dictionary mapping message types to Polars schemas.
    """
    plt.rcParams['text.usetex'] = True

    stab_thrust_df = deserialize_payload(df, "STAB_THRUST", schema)
    rotorcraft_status_df = deserialize_payload(df, "ROTORCRAFT_STATUS", schema)

    fig, axs = plt.subplots(2, 1, sharex=True)

    plot_thrust(stab_thrust_df, rotorcraft_status_df, axs[0])
    plot_thrust_rate(stab_thrust_df, rotorcraft_status_df, axs[1])

