import polars as pl
from typing import Dict, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background


def plot_control_allocation(df: pl.DataFrame, schema: Dict[str, pl.Schema], field_value: Optional[str] = None):
    plt.rcParams['text.usetex'] = True
    wls_v_df = deserialize_payload(df, "WLS_V", schema)
    wls_u_df = deserialize_payload(df, "WLS_U", schema)

    if field_value is not None:
        wls_v_df = wls_v_df.filter(pl.col("loop") == field_value)
        wls_u_df = wls_u_df.filter(pl.col("loop") == field_value)

    timestamp = wls_v_df["msg_timestamp"].to_numpy()
    v = wls_v_df["v"].to_numpy()
    v_cost = wls_v_df["Wv"].to_numpy()
    v_iter = wls_v_df["iter"].to_numpy()
    v_gamma = wls_v_df["gamma"].to_numpy()

    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plot_signals(v, timestamp, colors=list(["red", "green", "mediumblue", "orange"]),
                 signal_names=list(["v_roll", "v_pitch", "v_yaw", "v_thrust"]),
                 linestyle="dashed",
                 discrete=True,
                 where="post",
                 marker=".",
                 ax=axs[0])
    
    timestamp_u = wls_u_df["msg_timestamp"].to_numpy()
    u = wls_u_df["u"].to_numpy()
    u_pref = wls_u_df["u_pref"].to_numpy()
    u_min = wls_u_df["u_min"].to_numpy()
    u_max = wls_u_df["u_max"].to_numpy()
    u_cost = wls_u_df["Wu"].to_numpy()

    u_norm = u / np.max(u, axis=0)

    plot_signals(u_norm, timestamp_u, colors=["red", "green", "darkblue", "orange"],
                 signal_names=["ele_left", "ele_right", "motor_2_left", "motor_2_right"],
                 linestyle="solid",
                 discrete=False,
                 ax=axs[1])
    axs[0].set_ylabel("Control allocation output v")
    axs[0].legend(loc="lower left")
    axs[1].set_ylabel("Actuator commands u")
    axs[1].legend(loc="lower left")
    plt.tight_layout()

    nu_df = deserialize_payload(df, "EFF_MAT_STAB", schema)

    # Get control effectiveness matrix
    ce_roll = nu_df["G1_roll"].to_numpy()
    ce_pitch = nu_df["G1_pitch"].to_numpy()
    ce_yaw = nu_df["G1_yaw"].to_numpy()
    ce_thrust = nu_df["G1_thrust"].to_numpy()

    ce_mat = np.stack([ce_roll, ce_pitch, ce_yaw, ce_thrust], axis=-1)

    ce_mat_cnst = ce_mat[-1,:,:]
    print(ce_mat_cnst)




    achieved_ce = np.einsum('kj,ik->ij', ce_mat_cnst, u)

    plot_signals(achieved_ce, timestamp_u, ax=axs[0], colors=["red", "green", "mediumblue", "orange"],
                 signal_names=["rec_v_roll", "rec_v_pitch", "rec_v_yaw", "rec_v_thrust"],
                 linestyle="solid",
                 marker=".",
                 where="post",
                 discrete=True)
    axs[0].legend()

