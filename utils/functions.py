import numpy as np
import scipy as sc


def compute_covariance_matrix(signal):
    """
    Compute the covariance matrix of the signal
    :param signal: 2D array of shape (num_samples, num_channels)
    :return: 2D array of shape (num_channels, num_channels)
    """
    return np.cov(signal)


def compute_steering_vector(array_geometry: str, num_sensors: int, wavelength: int, theta: float):
    """
    Compute the steering vector for a given array geometry and wavelength
    :param array_geometry: str
    :param num_sensors: int
    :param wavelength:
    :param theta: float
    :return: 1D array of shape (num_sensors, )
    """
    if array_geometry == 'ULA':
        return np.exp(-1j * 2 * np.pi * np.arange(num_sensors) * np.sin(theta) / wavelength)
    else:
        raise ValueError('Invalid array geometry')

def find_spectrum_peaks(spectrum):
    """
    Find the indices of the peaks in the array
    :param array: 1D array
    :return: peaks indices
    """
    # Find spectrum peaks
    peaks = list(sc.signal.find_peaks(spectrum)[0])
    # Sort the peak by their amplitude
    peaks.sort(key=lambda x: spectrum[x], reverse=True)

    return peaks
