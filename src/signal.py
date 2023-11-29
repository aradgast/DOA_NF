import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from utils.functions import *


class Signal:

    def __init__(self, num_samples: int, num_sources: int, num_sensors: int, wavelength: int, array_geometry: str):
        self.num_samples = num_samples
        self.num_sources = num_sources
        self.num_sensors = num_sensors
        self.wavelength = wavelength
        self.array_geometry = array_geometry

    def generate(self, snr: float, angles: list):
        """
        Generate a signal with the given parameters
        :param snr: float
        :param angles: list
        :return: 2D array of shape (num_samples, num_sensors)
        """
        # Generate random noise
        noise = np.random.randn(self.num_samples, self.num_sensors)
        # Generate steering vectors
        steering_vectors = np.zeros((self.num_sensors, self.num_sources), dtype=complex)
        for i, theta in enumerate(angles):
            steering_vectors[:, i] = compute_steering_vector(self.array_geometry, self.num_sensors,
                                                             self.wavelength, theta)
        # Generate random source signals
        source_signals = np.random.randn(self.num_samples, self.num_sources)
        # Compute the signal
        signal = steering_vectors @ source_signals.T
        # Add noise
        noise_power = np.linalg.norm(signal) ** 2 / (self.num_samples * 10 ** (snr / 10))
        noise = np.sqrt(noise_power) * noise
        signal = signal + noise.T

        return signal
