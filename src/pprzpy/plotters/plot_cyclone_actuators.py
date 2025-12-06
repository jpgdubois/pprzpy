import polars as pl
from typing import Dict, List, Union
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, fill_signal_status_background, draw_signal_status_lines

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pprzpy.tools.display import plot_signals, fill_signal_status_background, draw_signal_status_lines


def plot_actuators_t4_out(
    actuators_df,
    ax: plt.Axes,
    servo_ids: List[int],
    esc_ids: List[int],
):
    """
    Plot selected servos and ESCs from actuator DataFrame on given axes.

    Args:
      actuators_df: Polars DataFrame with actuator signals
      axs: Tuple of two matplotlib Axes (first for servos, second for ESCs)
      servo_ids: List of servo numbers (1-12) to plot
      esc_ids: List of ESC numbers (13-16) to plot
      rotorcraft_status_df: Polars DataFrame with rotorcraft status signals for background
    """

    # Convert timestamps for x axis
    timestamps = actuators_df["msg_timestamp"].to_numpy()

    vals = actuators_df[f"esc_{esc_num}_dshot_cmd"].to_numpy() # IN a range of 0 - 2000 (typical for dshot commands)

    # Extract signals for requested servos
    servo_signals = np.vstack([actuators_df[f"servo_{sid}_angle_cmd"].to_numpy() * np.pi / 18000 for sid in servo_ids]).T
    servo_labels = [f"Servo {sid}" for sid in servo_ids]

    # Extract signals for requested ESCs
    esc_signals = np.vstack([actuators_df[f"esc_{eid}_dshot_cmd"].to_numpy() for eid in esc_ids]).T
    esc_labels = [f"ESC {eid}" for eid in esc_ids]

    plot_signals(servo_signals, timestamps,
                 signal_names=servo_labels,
                 colors=plt.cm.get_cmap('tab10').colors[:len(servo_ids)],
                 ax=axs[0])
    ax.set_ylabel(r"Servo $\textrm{rad}$")
    ax.legend(loc="lower left")

    # Plot ESC signals
    fill_signal_status_background(
        ap_mode, status_time, ax=axs[1],
        color_map={0: "red", 3: "yellow", 4: "green"},
        label_map={0: "Kill", 3: "Rate", 4: "Atti"},
        annotate=True, alpha=0.15)
    draw_signal_status_lines(
        ap_in_flight, status_time, ax=axs[1],
        label_map={0: "On Ground", 1: "In Flight"},
        annotate=True, alpha=1.0,
        linewidth=1.0, linestyle="dashed")
    plot_signals(esc_signals, timestamps,
                 signal_names=esc_labels,
                 colors=plt.cm.get_cmap('tab10').colors[:len(esc_ids)],
                 ax=axs[1])
    ax.set_ylabel(r"ESC $\frac{\textrm{rad}}{\textrm{s}}$")
    ax.legend(loc="lower left")




def plot_cyclone_actuators(df: pl.DataFrame, schema: Dict[str, pl.Schema]):

    plt.rcParams['text.usetex'] = True

    actuators_t4_in_df = deserialize_payload(df, "ACTUATORS_T4_IN", schema)
    actuators_t4_out_df = deserialize_payload(df, "ACTUATORS_T4_OUT", schema)

    rotorcraft_status_df = deserialize_payload(df, "ROTORCRAFT_STATUS", schema)

    actuators_t4_in_time = actuators_t4_in_df["msg_timestamp"].to_numpy()
    esc_1_in = actuators_t4_in_df["esc_1_rpm"].to_numpy() * 2 * np.pi / 60
    esc_2_in = actuators_t4_in_df["esc_2_rpm"].to_numpy() * 2 * np.pi / 60
    servo_1_in = actuators_t4_in_df["servo_1_angle"].to_numpy() * np.pi / 18000
    servo_6_in = -actuators_t4_in_df["servo_6_angle"].to_numpy() * np.pi / 18000

    actuators_t4_out_time = actuators_t4_out_df["msg_timestamp"].to_numpy()
    esc_1_out = actuators_t4_out_df["esc_1_dshot_cmd"].to_numpy()
    esc_2_out = actuators_t4_out_df["esc_2_dshot_cmd"].to_numpy()
    servo_1_out = actuators_t4_out_df["servo_1_angle_cmd"].to_numpy() * np.pi / 18000
    servo_6_out = -actuators_t4_out_df["servo_6_angle_cmd"].to_numpy() * np.pi / 18000

    rotorcraft_status_time = rotorcraft_status_df["msg_timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    fig, axs = plt.subplots(2, 1, sharex=True)

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=axs[0], color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=axs[0],
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    # Plot in servos (dashed)
    plot_signals(np.vstack([servo_1_in, servo_6_in]).T,
                 actuators_t4_in_time,
                 signal_names=["Elevon Left In", "Elevon Right In"],
                 colors=["blue", "orange"],
                 ax=axs[0],
                 linestyle='--')
    # Plot out servos (solid)
    plot_signals(np.vstack([servo_1_out, servo_6_out]).T,
                 actuators_t4_out_time,
                 signal_names=["Elevon Left Out", "Elevon Right Out"],
                 colors=["blue", "orange"],
                 ax=axs[0])
    axs[0].legend(loc="lower left")
    axs[0].set_ylabel(r"Servo $\textrm{rad}$")

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=axs[1], color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=axs[1],
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")
    # Plot in ESCs (dashed)
    plot_signals(np.vstack([esc_1_in, esc_2_in]).T,
                 actuators_t4_in_time,
                 signal_names=["Motor Left In", "Motor Right In"],
                 colors=["blue", "orange"],
                 ax=axs[1],
                 linestyle='--')
    # Plot out ESCs (solid)
    plot_signals(np.vstack([esc_1_out, esc_2_out]).T,
                 actuators_t4_out_time,
                 signal_names=["Motor Left Out", "Motor Right Out"],
                 colors=["blue", "orange"],
                 ax=axs[1])
    axs[1].set_xlabel(r"Time (\textrm{s})")
    axs[1].set_ylabel(r"ESC $\frac{\textrm{rad}}{\textrm{s}}$")
    axs[1].legend(loc="lower left")
