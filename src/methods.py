import numpy as np
import scipy as sc
from utils.functions import *


class MUSIC:
    def __init__(self, array_geometry: str, num_sensors: int, wavelength: int, num_sources=None):
        self.array_geometry = array_geometry
        self.wavelength = wavelength
        self.num_sensor = num_sensors
        self.num_sources = num_sources
        self.thera_range = np.linspace(-np.pi / 2, np.pi / 2, 18000, endpoint=False)

    def compute_predictions(self, signal, num_sources: int = None):
        """
        :param num_sources:
        :param signal:
        :return: DOAs
        """
        if num_sources is None:
            num_sources = self.num_sources
        cov_mat = compute_covariance_matrix(signal)
        eig_vals, eig_vecs = sc.linalg.eig(cov_mat)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        noise_eig_vecs = eig_vecs[:, num_sources:]

        music_spectrum = np.zeros(len(self.thera_range))
        for i, theta in enumerate(self.thera_range):
            steering_vec = compute_steering_vector(self.array_geometry, self.num_sensor, self.wavelength, theta)
            music_spectrum[i] = 1 / (np.linalg.norm(steering_vec.conj().T @ noise_eig_vecs) ** 2)

        peaks = find_spectrum_peaks(music_spectrum)
        peaks = np.array(peaks)
        predictions = self.thera_range[peaks][0:num_sources]
        # self.plot_spectrum(music_spectrum)
        return predictions

    def plot_spectrum(self, spectrum):
        plt.figure()
        plt.title("MUSIC spectrum")
        plt.plot(np.rad2deg(self.thera_range), spectrum)
        plt.grid()
        plt.show()


class MUSIC2D:
    def __init__(self, array_geometry: str, num_sensors: int, wavelength: int, num_sources: int = None):
        self.array_geometry = array_geometry
        self.wavelength = wavelength
        self.num_sensor = num_sensors
        self.num_sources = num_sources
        self.thera_range = np.linspace(0, np.pi, 360, endpoint=False)
        self.fraunhofer_distance = calculate_fraunhofer_distance(array_geometry, num_sensors, wavelength)
        self.distance_range = np.linspace(0.1 * self.fraunhofer_distance, 0.9 * self.fraunhofer_distance, 100,
                                          endpoint=False)

    def compute_predictions(self, signal):
        """
        :param signal:
        :return: DOAs
        """
        cov_mat = compute_covariance_matrix(signal)
        eig_vals, eig_vecs = sc.linalg.eig(cov_mat)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        noise_eig_vecs = eig_vecs[:, self.num_sources:]

        music_spectrum = np.zeros((len(self.thera_range), len(self.distance_range)))
        for idx_angle, theta in enumerate(self.thera_range):
            for idx_dist, dist in enumerate(self.distance_range):
                steering_vec = compute_steering_vector_2d(self.array_geometry, self.num_sensor, self.wavelength,
                                                          theta, dist)
                music_spectrum[idx_angle, idx_dist] = 1 / (np.linalg.norm(steering_vec.conj().T @ noise_eig_vecs) ** 2)

        peaks = find_spectrum_2d_peaks(music_spectrum)
        peaks = np.array(peaks)
        predict_theta = self.thera_range[peaks[0]][0:self.num_sources]
        predict_dist = self.distance_range[peaks[1]][0:self.num_sources]
        self.plot_spectrum(music_spectrum)
        return predict_theta, predict_dist

    def plot_spectrum(self, spectrum):
        # Creating figure
        x, y = np.meshgrid(self.distance_range, np.rad2deg(self.thera_range))
        # Plotting the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, spectrum, cmap='viridis')
        ax.set_title('MUSIC spectrum')
        ax.set_xlim(self.distance_range[0], self.distance_range[-1])
        ax.set_ylim(np.rad2deg(self.thera_range[0]), np.rad2deg(self.thera_range[-1]))
        # Adding labels
        ax.set_ylabel('Theta')
        ax.set_xlabel('Radius')
        ax.set_zlabel('Power')

        # Display the plot
        plt.show()


if __name__ == '__main__':
    pass
