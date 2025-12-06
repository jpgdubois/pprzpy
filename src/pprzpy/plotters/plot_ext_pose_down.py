import polars as pl
from typing import Dict, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pprzpy.tools.parse_data import deserialize_payload
from pprzpy.tools.display import plot_signals, draw_signal_status_lines, fill_signal_status_background, Condition, fill_signal_condition_background

def plot_ext_pose_down(df: pl.DataFrame, schema: Dict[str, pl.Schema], field_value: Optional[str] = None):
    plt.rcParams['text.usetex'] = True
    ext_pose_down_df = deserialize_payload(df, "EXTERNAL_POSE_DOWN", schema)
    print(ext_pose_down_df.tail())

    timestamp = ext_pose_down_df["msg_timestamp"].to_numpy()
    timestamp_ext = ext_pose_down_df["timestamp"].to_numpy()
    ned_x = ext_pose_down_df["ned_x"].to_numpy()
    ned_y = ext_pose_down_df["ned_y"].to_numpy()
    ned_z = ext_pose_down_df["ned_z"].to_numpy()
    ned = np.stack([ned_x, ned_y, ned_z], axis=1)

    body_qi = ext_pose_down_df["body_qi"].to_numpy()
    body_qx = ext_pose_down_df["body_qx"].to_numpy()
    body_qy = ext_pose_down_df["body_qy"].to_numpy()
    body_qz = ext_pose_down_df["body_qz"].to_numpy()
    body_quat = np.stack([body_qi, body_qx, body_qy, body_qz], axis=1)



    fig, axs = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
    plot_signals(body_quat, timestamp, colors=["red", "green", "blue", "orange"],
                 signal_names=["qi", "qx", "qy", "qz"],
                 linestyle="solid",
                 discrete=False,
                 marker=".",
                 ax=axs)

