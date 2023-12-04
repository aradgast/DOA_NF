import numpy as np
# import scipy as sc
# import matplotlib.pyplot as plt

from utils.functions import *


class Signal:

    def __init__(self, wavelength: int, array_geometry: str,
                 num_samples: int = None, num_sources: int = None, num_sensors: int = None):
        self.num_samples = num_samples
        self.num_sources = num_sources
        self.num_sensors = num_sensors
        self.wavelength = wavelength
        self.array_geometry = array_geometry

    def generate(self, snr: float, angles: list,
                 num_samples: int = None, num_sources: int = None, num_sensors: int = None):
        """
        Generate a signal with the given parameters
        :param snr: float
        :param angles: list
        :param num_samples: int
        :param num_sources: int
        :param num_sensors: int
        :return: 2D array of shape (num_samples, num_sensors)
        """
        if num_samples is None:
            num_samples = self.num_samples
        if num_sources is None:
            num_sources = self.num_sources
        if num_sensors is None:
            num_sensors = self.num_sensors
        # Generate random noise
        noise = (np.sqrt(2) / 2) * np.random.randn(num_samples, num_sensors) \
                + 1j * np.random.randn(num_samples, num_sensors)
        # Generate steering vectors
        steering_vectors = np.zeros((num_sensors, num_sources), dtype=complex)
        for i, theta in enumerate(angles):
            steering_vectors[:, i] = compute_steering_vector(self.array_geometry, num_sensors,
                                                             self.wavelength, theta)
        # Generate random source signals
        source_signals = 10 ** (snr / 10) * (np.sqrt(2) / 2) * np.random.randn(num_samples, num_sources) \
                         + 1j * np.random.randn(num_samples, num_sources)
        # Compute the signal
        signal = steering_vectors @ source_signals.T
        # Add noise
        signal = signal + noise.T

        return signal

    def generate_2d(self, snr: float, angles: list, distances: list,
                    num_samples: int = None, num_sources: int = None, num_sensors: int = None):
        """
        Generate a signal with the given parameters
        :param distances:
        :param snr: float
        :param angles: list
        :param num_samples: int
        :param num_sources: int
        :param num_sensors: int
        :return: 2D array of shape (num_samples, num_sensors)
        """
        if num_samples is None:
            num_samples = self.num_samples
        if num_sources is None:
            num_sources = self.num_sources
        if num_sensors is None:
            num_sensors = self.num_sensors
        # Generate random noise
        noise = (np.sqrt(2) / 2) * np.random.randn(num_samples, num_sensors) \
                + 1j * np.random.randn(num_samples, num_sensors)
        # Generate steering vectors
        steering_vectors = np.zeros((num_sensors, num_sources), dtype=complex)
        for i, (theta, dist) in enumerate(zip(angles, distances)):
            steering_vectors[:, i] = compute_steering_vector_2d(self.array_geometry, num_sensors,
                                                                self.wavelength, theta, dist)
        # Generate random source signals
        source_signals = 10 ** (snr / 10) * (np.sqrt(2) / 2) * np.random.randn(num_samples, num_sources) \
                         + 1j * np.random.randn(num_samples, num_sources)
        # Compute the signal
        signal = steering_vectors @ source_signals.T
        # Add noise
        signal = signal + noise.T

        return signal
