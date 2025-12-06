import numpy as np
from scipy.signal import lombscargle
from typing import Sequence, Tuple, Union

def calc_signal_spectral_density(
    timesteps: Union[Sequence[float], np.ndarray],
    signal: Union[Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectral density of a signal with associated non-uniform timesteps
    using the Lomb-Scargle periodogram.

    Parameters:
    - timesteps : 1D array-like of time points (non-uniform or uniform)
    - signal    : 1D array-like of signal values sampled at the given timesteps

    Returns:
    - freqs     : 1D numpy array of frequencies at which the PSD was computed (Hz)
    - psd       : 1D numpy array of power spectral density values (arbitrary units)
    """
    # Convert inputs to numpy arrays
    timesteps = np.asarray(timesteps, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if timesteps.ndim != 1 or signal.ndim != 1:
        raise ValueError("timesteps and signal must be 1D arrays")
    if len(timesteps) != len(signal):
        raise ValueError("timesteps and signal must have the same length")
    if len(timesteps) < 2:
        raise ValueError("need at least two samples to compute a periodogram")

    # Define frequency range for periodogram calculation
    # Use an evenly spaced frequency grid from near zero to Nyquist frequency
    nsamples = len(timesteps)
    duration = timesteps[-1] - timesteps[0]
    if duration <= 0:
        raise ValueError("timesteps must be strictly increasing")
    nyquist_freq = 0.5 * nsamples / duration
    freqs = np.linspace(0.01, nyquist_freq, 1000)

    # Angular frequencies for lombscargle
    angular_freqs = 2 * np.pi * freqs

    # Compute Lomb-Scargle periodogram
    pgram = lombscargle(timesteps, signal, angular_freqs)

    # Normalize power so that it is comparable to traditional PSD estimates
    psd = np.sqrt(4 * pgram / nsamples)

    return freqs, psd

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    # Create a sample signal with non-uniform timesteps
    np.random.seed(0)
    timesteps = np.sort(np.random.uniform(0, 10, 100))
    signal = np.sin(2 * np.pi * 1.0 * timesteps) + 0.5 * np.random.normal(size=timesteps.shape)

    # Calculate PSD
    freqs, psd = calc_signal_spectral_density(timesteps, signal)

    # Plot the results
    plt.figure()
    plt.semilogy(freqs, psd)
    plt.title("Power Spectral Density (Lomb-Scargle)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (arbitrary units)")
    plt.grid()
    plt.show()