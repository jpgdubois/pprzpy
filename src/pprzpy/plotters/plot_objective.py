import polars as pl
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background


def plot_objective(df: pl.DataFrame, schema: Dict[str, pl.Schema]):
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
    nu_df = deserialize_payload(df, "EFF_MAT_STAB", schema)
    actuators_df = deserialize_payload(df, "ACTUATORS_T4_IN", schema)

    # rotorcraft_status_df = deserialize_payload(df, "ROTORCRAFT_STATUS", schema)


    # Get control effectiveness matrix
    ce_roll = nu_df["G1_roll"].to_numpy()
    ce_pitch = nu_df["G1_pitch"].to_numpy()
    ce_yaw = nu_df["G1_yaw"].to_numpy()
    ce_thrust = nu_df["G1_thrust"].to_numpy()

    ce_mat = np.stack([ce_roll, ce_pitch, ce_yaw, ce_thrust], axis=-1)

    ce_timestamp = nu_df["msg_timestamp"].to_numpy()

    # Get actuator state
    actuator_elevon_left = actuators_df["servo_1_angle"].to_numpy() * np.pi / 18000
    actuator_elevon_right = actuators_df["servo_6_angle"].to_numpy() * np.pi / 18000
    actuator_motor_2_left = (actuators_df["esc_1_rpm"].to_numpy() * (2 * np.pi / 60))**2
    actuator_motor_2_right = (actuators_df["esc_2_rpm"].to_numpy() * (2 * np.pi / 60))**2

    actuator_state = np.stack([actuator_elevon_left, actuator_elevon_right,
                               actuator_motor_2_left, actuator_motor_2_right], axis=-1)

    actuator_timestamp = actuators_df["msg_timestamp"].to_numpy()

    # Select matching actuator state for each effectiveness matrix entry (pick closest in time)
    matched_actuator_state = []
    for t in ce_timestamp:
        time_diffs = np.abs(actuator_timestamp - t)
        closest_idx = np.argmin(time_diffs)
        matched_actuator_state.append(actuator_state[closest_idx])
    matched_actuator_state = np.array(matched_actuator_state)

    bandwidth = np.array([1/20.0, 1/20.0, 1/35.0, 1/35.0])  # Bandwidth values for actuators

    # Do CE_mat * bandwidth diagonal matrix * actuator_state
    achieved_ce = np.einsum('ikj,j,ik->ij', ce_mat, bandwidth, matched_actuator_state)
    # achieved_ce = np.einsum('ijk,ik->ij', ce_mat, matched_actuator_state)




    fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    plot_signals(achieved_ce, ce_timestamp, ax=axs, colors=["red", "green", "blue", "black"],
                 signal_names=["roll_effort", "pitch_effort", "yaw_effort", "thrust_effort"])
    axs.set_ylabel("Achieved control effort")
    axs.legend()

