import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background


def plot_stab_pseudo_command(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    """
    Wrapper function to deserialize data and plot pseudo command signals.

    Parameters
    ----------
    df : pl.DataFrame
        Raw telemetry data.
    schema : Dict[str, pl.Schema]
        Dictionary mapping message types to Polars schemas.
    """
    plt.rcParams['text.usetex'] = True

    pseudo_command_df = deserialize_payload(df, "STAB_PSEUDO_COMMAND", schema)
    rotorcraft_status_df = deserialize_payload(df, "ROTORCRAFT_STATUS", schema)

    timestamp = pseudo_command_df["msg_timestamp"].to_numpy()
    nu_obj = pseudo_command_df["nu_obj"].to_numpy()
    nu_ec = pseudo_command_df["nu_ec"].to_numpy()
    nu_obm = pseudo_command_df["nu_obm"].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    plot_signals(nu_obj, timestamp, ax=ax, colors=["red", "green", "darkblue", "orange"],
                 signal_names=["nu_roll_obj", "nu_pitch_obj", "nu_yaw_obj", "nu_thrust_obj"], linestyle="dashed", discrete=True)
    plot_signals(nu_ec, timestamp, ax=ax, colors=["red", "green", "darkblue", "orange"],
                 signal_names=["nu_roll_ec", "nu_pitch_ec", "nu_yaw_ec", "nu_thrust_ec"])
    plot_signals(nu_obm, timestamp, ax=ax, colors=["red", "green", "darkblue", "orange"],
                 signal_names=["nu_roll_obm", "nu_pitch_obm", "nu_yaw_obm", "nu_thrust_obm"])
    ax.set_ylabel("Pseudo commands")
    ax.legend(loc="lower left")


    plt.tight_layout()