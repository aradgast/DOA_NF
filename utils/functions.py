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


def calculate_fraunhofer_distance(array_geomatry: str, num_sensors: int, wavelenght: int):
    if array_geomatry == "ULA":
        D = (num_sensors - 1) * wavelenght / 2
        d_f = 2 * (D ** 2) / wavelenght
        return d_f, D
    else:
        raise TypeError(f"{array_geomatry} not supported")


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
        return np.exp(-1j * np.pi * np.linspace(0, num_sensors, num_sensors, endpoint=False)
                      * np.sin(theta) / wavelength)
    else:
        raise ValueError('Invalid array geometry')


def compute_steering_vector_2d(array_geometry: str, num_sensors: int, wavelength: int, theta: float, dist: float):
    """
    Compute the steering vector for a given array geometry and wavelength
    :param dist:
    :param array_geometry: str
    :param num_sensors: int
    :param wavelength:
    :param theta: float
    :return: 1D array of shape (num_sensors, )
    """
    if array_geometry == 'ULA':
        array = np.linspace(-num_sensors//2, num_sensors//2, num_sensors)
        first_order = array * np.sin(theta)
        second_order = -0.5 * (array * np.cos(theta)) ** 2 / dist

        return np.exp(-1j * 2 * np.pi * (first_order + second_order) / wavelength)
    else:
        raise ValueError('Invalid array geometry')


def find_spectrum_peaks(spectrum):
    """
    Find the indices of the peaks in the array
    :param spectrum:
    :return: peaks indices
    """
    # Find spectrum peaks
    peaks = list(sc.signal.find_peaks(spectrum)[0])
    # Sort the peak by their amplitude
    peaks.sort(key=lambda x: spectrum[x], reverse=True)

    return peaks


def find_spectrum_2d_peaks(spectrum):
    """
    Find the indices of the peaks in the array
    :param spectrum:
    :return: peaks indices
    """
    # Flatten the spectrum
    spectrum_flatten = spectrum.flatten()
    # Find spectrum peaks
    peaks = list(sc.signal.find_peaks(spectrum_flatten)[0])
    # Sort the peak by their amplitude
    peaks.sort(key=lambda x: spectrum_flatten[x], reverse=True)
    # convert the peaks to 2d indices
    original_idx = np.unravel_index(peaks, spectrum.shape)

    return list(original_idx)


def choose_angles(num_angles: int, angle_low: float = -90, angle_high: float = 90, min_gap: int = 2,
                  max_gap: int = 180):
    """
    Choose random angles
    :param max_gap:
    :param num_angles: int
    :param angle_low: float
    :param angle_high: float
    :param min_gap: int
    :return: list of angles
    """
    angles = []
    while len(angles) < num_angles:
        angle = np.random.randint(angle_low, angle_high)
        if len(angles) == 0:
            angles.append(angle)
        else:
            if np.min(np.abs(np.array(angles) - angle)) >= min_gap and \
                    np.max(np.abs(np.array(angles) - angle)) <= max_gap:
                angles.append(angle)
    return np.deg2rad(angles)


def choose_distances(num_distances: int, array_geomatry: str, num_sensors: int, wavelength: int, min_gap: int = 1,
                     max_gap: int = 100):
    """
    Choose random distances
    :param wavelength:
    :param num_sensors:
    :param array_geomatry:
    :param num_distances:
    :param min_gap:
    :param max_gap:
    :return: distances list
    """
    distance_high, distance_low = calculate_fraunhofer_distance(array_geomatry, num_sensors, wavelength)
    distances = []
    while len(distances) < num_distances:
        distance = np.random.randint(distance_low, distance_high)
        if len(distances) == 0:
            distances.append(distance)
        else:
            if np.min(np.abs(np.array(distances) - distance)) >= min_gap and \
                    np.max(np.abs(np.array(distances) - distance)) <= max_gap:
                distances.append(distance)
    return distances


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
