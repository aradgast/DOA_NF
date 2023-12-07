import numpy as np
import scipy as sc
from utils.functions import *
from src.modules import Module


class MUSIC:
    def __init__(self, module: Module, num_sources=None):
        self.module = module
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
            steering_vec = self.module.compute_steering_vector(theta)
            music_spectrum[i] = 1 / (np.linalg.norm(steering_vec.conj().T @ noise_eig_vecs) ** 2)

        peaks = self.find_spectrum_peaks(music_spectrum)
        peaks = np.array(peaks)
        predictions = self.thera_range[peaks][0:num_sources]
        if len(predictions) != num_sources:
            tmp = np.mean(predictions)
            predictions = list(predictions)
            for i in range(num_sources-len(predictions)):
                predictions.append(tmp)
            predictions = np.array(predictions)
        # self.plot_spectrum(music_spectrum)
        return predictions

    def plot_spectrum(self, spectrum):
        plt.figure()
        plt.title("MUSIC spectrum")
        plt.plot(np.rad2deg(self.thera_range), spectrum)
        plt.grid()
        plt.show()

    def find_spectrum_peaks(self, spectrum):
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


class MUSIC2D:
    def __init__(self, module: Module, num_sources: int = None):
        self.module = module
        self.num_sources = num_sources
        self.thera_range = np.arange(-np.pi / 2, np.pi / 2, np.pi/1800)
        self.fraunhofer_distance, D = self.module.calculate_fraunhofer_distance()
        self.distance_range = np.linspace(D, self.fraunhofer_distance, 1800)
        # print(f"fraunhofer_dist = {self.fraunhofer_distance}, D = {D}")

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
        noise_eig_vecs = eig_vecs[:, self.num_sources:]

        # music_spectrum = np.zeros((len(self.thera_range), len(self.distance_range)))
        # for idx_angle, theta in enumerate(self.thera_range):
        #     for idx_dist, dist in enumerate(self.distance_range):
        #         steering_vec = self.module.compute_steering_vector(theta, dist)
        #         steering_vec = np.squeeze(steering_vec)
        #         inverse_spectrum = np.real(steering_vec.conj().T @ noise_eig_vecs @ noise_eig_vecs.conj().T @ steering_vec)
        #         music_spectrum[idx_angle, idx_dist] = 1 / inverse_spectrum

        steering_vec = self.module.compute_steering_vector(self.thera_range, self.distance_range)
        var_1 = np.einsum("ijk,kl->ijl", np.transpose(steering_vec.conj(), (2, 1, 0)), noise_eig_vecs)
        var_2 = np.transpose(var_1.conj(), (2, 1, 0))
        inverse_spectrum = np.real(np.einsum("ijk,kji->ji",var_1, var_2))
        music_spectrum = 1 / inverse_spectrum


        peaks = self.find_spectrum_peaks(music_spectrum)
        peaks = np.array(peaks)
        predict_theta = self.thera_range[peaks[0]][0:self.num_sources]
        predict_dist = self.distance_range[peaks[1]][0:self.num_sources]
        # self.plot_heatmap(music_spectrum)
        self.plot_3d_spectrum(music_spectrum)
        return predict_theta, predict_dist

    def plot_heatmap(self, spectrum):
        data = np.log1p(spectrum)
        plt.figure()
        plt.title("MUSIC spectrum")
        plt.imshow(data, cmap='viridis', aspect='auto', origin='lower',
                   extent=[min(self.distance_range), max(self.distance_range),
                           min(np.rad2deg(self.thera_range)), max(np.rad2deg(self.thera_range))])
        plt.colorbar()
        plt.xlabel('Distance')
        plt.ylabel('Theta')
        plt.grid()
        plt.show()

    def plot_3d_spectrum(self, spectrum):
        # Creating figure
        x, y = np.meshgrid(self.distance_range, np.rad2deg(self.thera_range))
        # Plotting the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, np.log1p(spectrum), cmap='viridis')
        ax.set_title('MUSIC spectrum')
        ax.set_xlim(self.distance_range[0], self.distance_range[-1])
        ax.set_ylim(np.rad2deg(self.thera_range[0]), np.rad2deg(self.thera_range[-1]))
        # Adding labels
        ax.set_ylabel('Theta')
        ax.set_xlabel('Radius')
        ax.set_zlabel('Power')

        # Display the plot
        plt.show()

    def find_spectrum_peaks(self, spectrum):
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


if __name__ == '__main__':
    pass
