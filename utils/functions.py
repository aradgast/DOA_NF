import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


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
        return np.exp(-1j * 2 * np.pi * np.linspace(0, num_sensors, num_sensors, endpoint=False) * np.sin(theta) / wavelength)
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


def plot_angles_on_unit_circle(true, predections):
    # plot the angles theats and angles predictions on the half unit circle: -90 to 90 degrees
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.deg2rad(true), np.ones(len(true)), 'ro', label='true')
    ax.plot(np.deg2rad(predections), np.ones(len(predections)), 'bo', label='predicted')
    ax.set_title("Angles on unit circle", va='bottom')
    # ax.set_xticks(np.pi / 180. * np.linspace(-90, 90, 18, endpoint=False))
    # ax.set_xticklabels(np.linspace(-90, 90, 18, endpoint=False))
    ax.legend()
    plt.show()
