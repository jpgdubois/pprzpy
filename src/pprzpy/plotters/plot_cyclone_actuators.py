import polars as pl
from typing import Dict, List, Union
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, fill_signal_status_background, draw_signal_status_lines

def plot_cyclone_actuators(df: pl.DataFrame, schema: Dict[str, pl.Schema]):

    plt.rcParams['text.usetex'] = True

    actuators_t4_in_df = deserialize_payload(df, "ACTUATORS_T4_IN", schema)
    # actuators_t4_out_df = deserialize_payload(df, "ACTUATORS_T4_OUT", schema)

    rotorcraft_status_df = deserialize_payload(df, "ROTORCRAFT_STATUS", schema)

    actuators_t4_in_time = actuators_t4_in_df["timestamp"].to_numpy()
    esc_1_in = actuators_t4_in_df["esc_1_rpm"].to_numpy() * 2 * np.pi / 60
    esc_2_in = actuators_t4_in_df["esc_2_rpm"].to_numpy() * 2 * np.pi / 60
    servo_1_in = actuators_t4_in_df["servo_1_angle"].to_numpy() * np.pi / 18000
    servo_6_in = actuators_t4_in_df["servo_6_angle"].to_numpy() * np.pi / 18000

    rotorcraft_status_time = rotorcraft_status_df["timestamp"].to_numpy()
    ap_mode = rotorcraft_status_df["ap_mode"].to_numpy()
    ap_in_flight = rotorcraft_status_df["ap_in_flight"].to_numpy()

    # actuators_t4_out_time = actuators_t4_out_df["timestamp"].to_numpy()
    fig, axs = plt.subplots(2, 1, sharex=True)

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=axs[0], color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=axs[0],
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")

    plot_signals(np.vstack([servo_1_in, servo_6_in]).T,
                 actuators_t4_in_time,
                 signal_names=["Elevon Left", "Elevon Right"],
                 colors=["blue", "orange"],
                 ax=axs[0])
    axs[0].legend(loc="lower left")
    axs[0].set_ylabel(r"Servo $\textrm{rad}$")

    fill_signal_status_background(ap_mode, rotorcraft_status_time, ax=axs[1], color_map={0: "red", 3: "yellow", 4: "green"},
                                  label_map={0: "Kill", 3: "Rate", 4: "Atti"}, annotate=True, alpha=0.15)
    draw_signal_status_lines(ap_in_flight, rotorcraft_status_time, ax=axs[1],
                             label_map={0: "On Ground", 1: "In Flight"}, annotate=True, alpha=1.0,
                             linewidth=1.0, linestyle="dashed")
    plot_signals(np.vstack([esc_1_in, esc_2_in]).T,
                 actuators_t4_in_time,
                 signal_names=["Motor Left", "Motor Right"],
                 colors=["blue", "orange"],
                 ax=axs[1])
    axs[1].set_xlabel(r"Time (\textrm{s})")
    axs[1].set_ylabel(r"ESC $\frac{\textrm{rad}}{\textrm{s}}$")
    axs[1].legend(loc="lower left")

