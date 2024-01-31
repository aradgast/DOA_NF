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
            for i in range(num_sources - len(predictions)):
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
        self.thera_range = np.arange(-np.pi / 2, np.pi / 2, np.pi / 90)
        self.fraunhofer_distance, self.fersnel_distance = self.module.calculate_fraunhofer_distance()
        print(self.module.calculate_fraunhofer_distance())
        self.distance_range = np.arange(self.fersnel_distance, self.fraunhofer_distance, 1)
        self.grid = self.module.compute_steering_vector(self.thera_range, self.distance_range)
        # print(f"fraunhofer_dist = {self.fraunhofer_distance}, D = {D}")

    def compute_predictions(self, signal, num_sources: int = None, threshold: int = None,
                            plot_spectrum: bool = False, soft_decsicion: bool = False):
        """
        :param threshold:
        :param num_sources:
        :param signal:
        :return: DOAs
        """
        if num_sources is None:
            num_sources = self.num_sources
        cov_mat = compute_covariance_matrix(signal)
        eig_vals, eig_vecs = sc.linalg.eig(cov_mat)
        if not threshold is None:
            mask = [eig_vals > threshold]
            num_sources = np.sum(mask)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        noise_eig_vecs = eig_vecs[:, num_sources:]

        var_1 = np.einsum("ijk,kl->ijl", np.transpose(self.grid.conj(), (2, 1, 0)), noise_eig_vecs)
        var_2 = np.transpose(var_1.conj(), (2, 1, 0))
        inverse_spectrum = np.real(np.einsum("ijk,kji->ji", var_1, var_2))
        music_spectrum = 1 / inverse_spectrum

        if soft_decsicion:
            predict_theta, predict_dist = self.maskpeaks(music_spectrum, num_sources)
        else:
            peaks = self.find_spectrum_peaks(music_spectrum)
            peaks = np.array(peaks)
            predict_theta = self.thera_range[peaks[0]][0:num_sources]
            predict_dist = self.distance_range[peaks[1]][0:num_sources]

        if plot_spectrum:
            self.plot_3d_spectrum(music_spectrum)

        return predict_theta, predict_dist

    def plot_3d_spectrum(self, spectrum, highlight_coordinates=None):
        # Creating figure
        x, y = np.meshgrid(self.distance_range, np.rad2deg(self.thera_range))
        # Plotting the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, np.log1p(spectrum), cmap='viridis')

        if highlight_coordinates:
            highlight_coordinates = np.array(highlight_coordinates)
            ax.scatter(
                highlight_coordinates[:, 0],
                np.rad2deg(highlight_coordinates[:, 1]),
                np.log1p(highlight_coordinates[:, 2]),
                color='red',
                s=50,
                label='Highlight Points'
            )
        ax.set_title('MUSIC spectrum')
        ax.set_xlim(self.distance_range[0], self.distance_range[-1])
        ax.set_ylim(np.rad2deg(self.thera_range[0]), np.rad2deg(self.thera_range[-1]))
        # Adding labels
        ax.set_ylabel('Theta')
        ax.set_xlabel('Radius')
        ax.set_zlabel('Power')

        if highlight_coordinates:
            ax.legend() # Adding a legend

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

    def maskpeaks(self, spectrum: np.ndarray, P: int):

        top_indxs = np.argpartition(spectrum.reshape(1, -1).squeeze(), -P)[-P:]
        max_row = (np.floor(np.divide(top_indxs, spectrum.shape[1]))).astype(int)
        max_col = (top_indxs % spectrum.shape[1]).astype(int)
        soft_row = []
        soft_col = []
        cell_size = 20
        for i, (max_r, max_c) in enumerate(zip(max_row, max_col)):
            max_row_cell_idx = max_r - cell_size + \
                               np.arange(2 * cell_size + 1, dtype=int).reshape(-1, 1)
            max_row_cell_idx = max_row_cell_idx[max_row_cell_idx >= 0]
            max_row_cell_idx = max_row_cell_idx[max_row_cell_idx < spectrum.shape[0]].reshape(-1, 1)

            max_col_cell_idx = max_c - cell_size + \
                               np.arange(2 * cell_size + 1, dtype=int).reshape(1, -1)
            max_col_cell_idx = max_col_cell_idx[max_col_cell_idx >= 0]
            max_col_cell_idx = max_col_cell_idx[max_col_cell_idx < spectrum.shape[1]].reshape(1, -1)

            metrix_thr = spectrum[max_row_cell_idx, max_col_cell_idx]
            metrix_thr /= np.max(metrix_thr)
            soft_max = np.exp(metrix_thr) / np.sum(np.exp(metrix_thr))

            soft_row.append(self.thera_range[max_row_cell_idx].T @ np.sum(soft_max, axis=1))
            soft_col.append(self.distance_range[max_col_cell_idx] @ np.sum(soft_max, axis=0))

        return soft_row, soft_col

class ESPRIT:
    def __init__(self, module: Module, num_sources: int = None):
        self.module = module
        self.num_sources = num_sources
    def compute_predictions(self, signal, num_sources: int = None, threshold: int = None,
                            plot_spectrum: bool = False, soft_decsicion: bool = False):
        if num_sources is None:
            num_sources = self.num_sources
        # get the 3 estimated covariance matrices
        R_1, R_2, R_3 = self.__calculate_shifted_covariance(signal)
        # stack the 3 matrices together
        R = np.concatenate((R_1, R_2, R_3), axis=0)
        # SVD calculation
        U, S, Vh = np.linalg.svd(R, full_matrices=True)
        # sort the  by the singularvectors by the singularvalues
        U = U[:, np.argsort(S)[::-1]]
        # take the singular vectors that correspond to the num_sources biggest singular values.
        E_s = U[:, :num_sources]
        E_0 = E_s[:E_s.shape[0] // 3]
        E_1 = E_s[E_s.shape[0] // 3:2 * E_s.shape[0] // 3]
        E_2 = E_s[2 * E_s.shape[0] // 3:]
        pinv_E_0 = np.linalg.pinv(E_0)
        psi_1 = pinv_E_0 @ E_1
        psi_2 = pinv_E_0 @ E_2
        omega = 0.5 * np.angle(psi_1)
        phi = -0.5 * np.angle(psi_2)
        angles = np.arcsin(-4 * omega / (2 * np.pi))
        distances = np.pi * self.module.wavelength * (np.cos(angles) ** 2) / (16 * phi)

        return angles, distances

    def __calculate_shifted_covariance(self, signal):
        # for R1 the correlation is between the n-m sensor to the n-m-1 sensor
        R_1 = (signal[:-2, :] @ signal[1:-1, :].conj().T) / signal.shape[1]
        R_2 = (signal[1:-1, :] @ signal[2:, :].conj().T) / signal.shape[1]
        R_3 = R_2.conj().T

        return R_1, R_2, R_3

if __name__ == '__main__':
    pass
