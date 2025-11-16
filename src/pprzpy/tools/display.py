"""
Signal visualization utilities for printing and plotting time-series data with annotations
and discrete state highlighting.

This module provides functions to print signal arrays with optional timestamps, plot
multi-signal time series with labels and units, and annotate discrete state signals by
filling background regions or drawing vertical lines at state transitions, optionally
with labels.

Functions
---------
print_signal:
    Print an N×M numpy array signal as a formatted table, optionally with timestamps.
plot_signals:
    Plot multiple signals over time on a single Matplotlib Axes with labels, units, colors.
fill_signal_status_background:
    Fill background regions on an Axes to highlight discrete state regimes with optional labels.
draw_signal_status_lines:
    Draw vertical lines on an Axes at discrete state transitions with optional labels.
"""

__all__ = [
    "print_signal",
    "plot_signals",
    "fill_signal_status_background",
    "draw_signal_status_lines",
]

from collections.abc import Sequence, Mapping
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def print_signal(
    signal: NDArray[np.float64],
    signal_names: Sequence[str] | None = None,
    timestamps: Sequence[float] | None = None,
) -> None:
    """
    Print an N×M numpy array `signal` in a readable table format, optionally with
    timestamps and column labels.

    Parameters
    ----------
    signal : ndarray of shape (N, M)
        The signal to print.
    signal_names : sequence of str, optional
        List of column names. Defaults to "Signal 1", "Signal 2", etc.
    timestamps : sequence of float, optional
        Timestamps corresponding to each row. Printed above the values if provided.

    Raises
    ------
    ValueError
        If the length of `signal_names` or `timestamps` does not match the respective
        dimensions of `signal`.
    """
    signal = np.atleast_2d(signal)
    n_rows, n_cols = signal.shape

    # Default signal names
    if signal_names is None:
        signal_names = [f"Signal {i + 1}" for i in range(n_cols)]

    if len(signal_names) != n_cols:
        raise ValueError(
            f"Expected {n_cols} signal names, got {len(signal_names)}."
        )

    if timestamps is not None and len(timestamps) != n_rows:
        raise ValueError(
            f"Expected {n_rows} timestamps, got {len(timestamps)}."
        )

    # Prepare table rows
    rows = [[signal_names[i], *signal[:, i]] for i in range(n_cols)]
    ts_row = [""] + ([f"{t:.6g}" for t in timestamps] if timestamps is not None else [])

    # Compute column widths
    all_rows = [ts_row, *rows] if timestamps is not None else rows
    col_widths = [max(len(str(item)) for item in col) for col in zip(*all_rows)]

    # Print timestamps
    if timestamps is not None:
        ts_str = " | ".join(
            [
                ts_row[0].rjust(col_widths[0]),
                *[ts_row[i].ljust(col_widths[i]) for i in range(1, len(ts_row))],
            ]
        )
        print(ts_str)

    # Print signal rows
    for row in rows:
        signal_name = row[0].rjust(col_widths[0])
        values = [str(x).ljust(col_widths[i + 1]) for i, x in enumerate(row[1:])]
        print(" | ".join([signal_name, *values]))


def plot_signals(
    signal: NDArray[np.floating],
    timesteps: NDArray[np.floating],
    signal_names: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    colors: Sequence[str] | None = None,
    discrete: bool = False,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot multiple time signals on a single matplotlib Axes with optional labels and units.

    Parameters
    ----------
    signal : ndarray of shape (N, M)
        The signal values.
    timesteps : sequence of float
        Time values corresponding to each signal row.
    signal_names : sequence of str, optional
        Names of the signals.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Defaults to current axes.
    colors : sequence of str, optional
        Colors for each signal.
    discrete : bool, optional
        Use step plots instead of lines.
    plot_kwargs : dict, optional
        Extra arguments passed to `plot` or `step`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plotted signals.

    Raises
    ------
    ValueError
        If array lengths or numbers of signals do not match expected dimensions.
    """
    signal = np.atleast_2d(signal)
    n_rows, n_cols = signal.shape

    if len(timesteps) != n_rows:
        raise ValueError(
            f"Expected {n_rows} timesteps, got {len(timesteps)}."
        )

    if signal_names is None:
        signal_names = [f"Signal {i + 1}" for i in range(n_cols)]

    for name_list, expected, label in [
        (signal_names, n_cols, "signal names"),
        (colors, n_cols, "colors") if colors else (None, None, None),
    ]:
        if name_list is not None and len(name_list) != expected:
            raise ValueError(f"Expected {expected} {label}, got {len(name_list)}.")

    ax = ax or plt.gca()

    for i in range(n_cols):
        label = signal_names[i]
        plot_func = ax.step if discrete else ax.plot
        kwargs = {**plot_kwargs}
        if colors is not None:
            kwargs["color"] = colors[i]
        plot_func(timesteps, signal[:, i], label=label, **kwargs)

    return ax


def _annotate_status(
    ax: plt.Axes,
    timesteps: NDArray,
    region_starts: list[int],
    signal: NDArray,
    label_map: Optional[Mapping[int, str]],
    color_map: Optional[Mapping[int, str]],
    y_frac: float,
) -> None:
    """
    Annotate vertical text labels at specified regions on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    timesteps : ndarray
        Array of time values.
    region_starts : list of int
        Indices where annotations are placed.
    signal : ndarray
        Signal values.
    label_map : dict or None
        Mapping from signal values to labels.
    color_map : dict or None
        Mapping from signal values to colors.
    y_frac : float
        Vertical position fraction in axes coordinates for the text.
    """
    for idx in region_starts:
        state = signal[idx]
        color = color_map.get(state, "black") if color_map else "black"
        label = label_map.get(state, str(state)) if label_map else str(state)

        ax.annotate(
            label,
            xy=(timesteps[idx], y_frac),
            xycoords=("data", "axes fraction"),
            ha="left",
            va="top",
            fontsize="medium",
            color=color,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
        )


def fill_signal_status_background(
    signal: NDArray,
    timesteps: NDArray,
    color_map: Optional[Mapping[int, str]] = None,
    label_map: Optional[Mapping[int, str]] = None,
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
    **plot_kwargs
) -> plt.Axes:
    """
    Fill background color spans on an axis to indicate discrete signal states with optional labels.

    Parameters
    ----------
    signal : ndarray
        1D array of discrete state values.
    timesteps : ndarray
        1D array of time values.
    color_map : dict or None, optional
        Mapping from state values to colors. Defaults to white.
    label_map : dict or None, optional
        Mapping from state values to label strings.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. Defaults to current axis.
    annotate : bool, optional
        Whether to annotate filled regions with labels.
    plot_kwargs : dict, optional
        Additional keyword arguments passed to axvspan.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with background spans added.

    Raises
    ------
    ValueError
        If timesteps and signal shapes do not match.
    """
    timesteps = np.asarray(timesteps)
    signal = np.asarray(signal)

    if timesteps.shape != signal.shape:
        raise ValueError("timesteps and signal must have the same shape.")

    ax = ax or plt.gca()

    region_start = 0
    region_starts = []
    for idx in range(1, len(signal) + 1):
        if idx == len(signal) or signal[idx] != signal[region_start]:
            state = signal[region_start]
            color = color_map.get(state, "white") if color_map else "white"
            ax.axvspan(timesteps[region_start], timesteps[idx - 1], color=color, **plot_kwargs)
            region_starts.append(region_start)
            region_start = idx

    if annotate:
        _annotate_status(ax, timesteps, region_starts, signal, label_map, color_map, y_frac=1.0)

    return ax


def draw_signal_status_lines(
    signal: NDArray,
    timesteps: NDArray,
    color_map: Optional[Mapping[int, str]] = {},
    label_map: Optional[Mapping[int, str]] = None,
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
    **plot_kwargs
) -> plt.Axes:
    """
    Draw vertical lines on an axis at discrete state transitions with optional labels.

    Parameters
    ----------
    signal : ndarray
        1D array of discrete state values.
    timesteps : ndarray
        1D array of time values.
    color_map : dict or None, optional
        Mapping from state values to colors. Defaults to black.
    label_map : dict or None, optional
        Mapping from state values to label strings.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. Defaults to current axis.
    annotate : bool, optional
        Whether to label vertical lines with state names.
    plot_kwargs : dict, optional
        Additional keyword arguments passed to axvline.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with vertical lines added.

    Raises
    ------
    ValueError
        If timesteps and signal shapes do not match.
    """
    timesteps = np.asarray(timesteps)
    signal = np.asarray(signal)

    if timesteps.shape != signal.shape:
        raise ValueError("timesteps and signal must have the same shape.")

    ax = ax or plt.gca()

    region_start = 0
    region_starts = []
    for idx in range(1, len(signal)):
        if signal[idx] != signal[region_start]:
            color = color_map.get(signal[region_start], "black") if color_map else "black"
            ax.axvline(timesteps[region_start], color=color, **plot_kwargs)
            region_starts.append(region_start)
            region_start = idx
    # Draw last line at final region start
    color = color_map.get(signal[region_start], "black") if color_map else "black"
    ax.axvline(timesteps[region_start], color=color, **plot_kwargs)
    region_starts.append(region_start)

    if annotate:
        _annotate_status(ax, timesteps, region_starts, signal, label_map, color_map, y_frac=0.95)

    return ax