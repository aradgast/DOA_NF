import numpy as np
import scipy as sc
from utils.functions import *


class MUSIC:
    def __init__(self, array_geometry: str, num_sensors: int, wavelength: int, num_sources=None):
        self.array_geometry = array_geometry
        self.wavelength = wavelength
        self.num_sensor = num_sensors
        self.num_sources = num_sources

    def compute(self, signal):
        """
        :param signal:
        :return: DOAs
        """
        cov_mat = compute_covariance_matrix(signal)
        eig_vals, eig_vecs = sc.linalg.eig(cov_mat)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        noise_eig_vecs = eig_vecs[:, self.num_sources:]

        thera_range = np.linspace(0, np.pi, 18000, endpoint=False)
        music_spectrum = np.zeros(len(thera_range))
        for i, theta in enumerate(thera_range):
            steering_vec = compute_steering_vector(self.array_geometry, self.num_sensor, self.wavelength, theta)
            music_spectrum[i] = 1 / (np.linalg.norm(steering_vec.conj().T @ noise_eig_vecs) ** 2)

        peaks = find_spectrum_peaks(music_spectrum)
        predictions = np.rad2deg(thera_range[peaks])[0:self.num_sources]

        return predictions[::-1]


if __name__ == '__main__':
    pass
