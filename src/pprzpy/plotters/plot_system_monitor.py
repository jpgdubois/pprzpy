import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background


def plot_system_monitor(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
    """
    Wrapper function to deserialize data and plot system monitor signals.

    Parameters
    ----------
    df : pl.DataFrame
        Raw telemetry data.
    schema : Dict[str, pl.Schema]
        Dictionary mapping message types to Polars schemas.
    """
    plt.rcParams['text.usetex'] = True

    system_monitor_df = deserialize_payload(df, "SYS_MON", schema)
    if system_monitor_df.is_empty():
        return

    timestamp = system_monitor_df["msg_timestamp"].to_numpy()
    cpu_load = system_monitor_df["cpu_load"].to_numpy()
    cpu_time = system_monitor_df["cpu_time"].to_numpy()
    periodic_time = system_monitor_df["periodic_time"].to_numpy()
    periodic_time_max = system_monitor_df["periodic_time_max"].to_numpy()
    periodic_time_min = system_monitor_df["periodic_time_min"].to_numpy()
    periodic_cycle = system_monitor_df["periodic_cycle"].to_numpy()
    periodic_cycle_max = system_monitor_df["periodic_cycle_max"].to_numpy()
    periodic_cycle_min = system_monitor_df["periodic_cycle_min"].to_numpy()

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plot_signals(cpu_load, timestamp, colors=["red"], signal_names=["CPU Load (%)"], linestyle="solid", discrete=False, ax=ax[0])