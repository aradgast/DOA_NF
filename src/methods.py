import numpy as np
import scipy as sc
from utils.functions import *
from src.modules import Module
from scipy.optimize import minimize


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
        self.thera_range = np.arange(-np.pi / 2, np.pi / 2, np.pi / 1800)
        self.fraunhofer_distance, self.D = self.module.calculate_fraunhofer_distance()
        self.distance_range = np.arange(self.module.wavelength, 30, 0.01)
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
            ax.legend()  # Adding a legend

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
    """

    """

    def __init__(self, module: Module, num_sources: int = None, shift: int = 1):
        self.shift = shift
        self.module = module
        self.num_sources = num_sources
        fraunhofer, fresnel = self.module.calculate_fraunhofer_distance()
        self.distance_range = np.arange(fresnel, fraunhofer, 0.01)

    def compute_predictions(self, signal: np.ndarray, num_sources: int = None):
        """

        :param signal:
        :param num_sources:
        :return:
        """
        cov_mat = compute_covariance_matrix(signal)
        eig_vals, eig_vecs = sc.linalg.eig(cov_mat)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        signal_eig_vec = eig_vecs[:, :num_sources]
        u_s_1, u_s_2 = signal_eig_vec[:signal_eig_vec.shape[0] - self.shift], signal_eig_vec[
                                                                              self.shift:signal_eig_vec.shape[0]]
        phi = np.linalg.pinv(u_s_1) @ u_s_2
        phi_eigenvalues = np.linalg.eigvals(phi)
        doa = -1 * np.arcsin(np.imag((1 / self.shift) * np.log(phi_eigenvalues)) / np.pi)

        distances = np.zeros(len(doa))
        # calculate grid for each source
        # for each angle in know_angle we need to calculate the spectrum
        noise_eig_vec = eig_vecs[:, num_sources:]
        limit = self.module.num_sensors // 2
        array = np.linspace(-limit, limit, self.module.num_sensors)
        for k, theta_k in enumerate(doa):
            music_spectrum = np.zeros(len(self.distance_range))
            for i, r in enumerate(self.distance_range):
                first_order = np.sin(theta_k) * array
                second_order = - 0.5 * np.power(np.cos(theta_k) * array / r, 2)
                time_delay = first_order + second_order
                A = np.exp(-1j * 4 * np.pi * time_delay / self.module.wavelength)[:, np.newaxis]
                music_spectrum[i] = 1 / (np.linalg.norm(A.conj().T @ noise_eig_vec) ** 2)
            print(music_spectrum.shape)
            plt.plot(self.distance_range, music_spectrum)
            plt.show()
            # idx = np.argmax(music_spectrum)
            # Find spectrum peaks
            peaks = list(sc.signal.find_peaks(music_spectrum)[0])
            # Sort the peak by their amplitude
            peaks.sort(key=lambda x: music_spectrum[x], reverse=True)
            idx = peaks[0]
            prediction = self.distance_range[idx]
            distances[k] = prediction

        return doa, distances

class RoootMusic:
    """

    """

    def __init__(self, module: Module):
        self.wavelength = module.wavelength
        self.gen_mat = None

    def compute_predictions(self, signal: np.ndarray, num_sources: int):
        """

        :param signal:
        :param num_sources:
        :return:
        """
        cov_mat = compute_covariance_matrix(signal)
        eig_vals, eig_vecs = sc.linalg.eig(cov_mat)
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)[::-1]]
        noise_eig_vecs = eig_vecs[:, num_sources:]
        self.gen_mat = noise_eig_vecs @ np.conj(noise_eig_vecs).T
        # calculate the sum of all diagonals in F -> coeffcients of the polynom
        coeff = self.calc_p_polynum_coeff()
        roots = self.find_roots(coeff)
        local_theta = np.ones((1, 2))
        local_values = np.array([])
        for k in range(roots.shape[0]):
            tmp_thetas, tmp_values = self.theta_prediction_loop(roots[k])
            if tmp_thetas is None:
                continue
            local_values = np.append(local_values, tmp_values)
            if k == 0:
                local_theta = tmp_thetas
            else:
                local_theta = np.concatenate((local_theta, tmp_thetas), axis=0)

        global_theta = self.find_global_theta(local_theta, local_values, num_sources)
        doa, ranges = self.compute_doa_range(global_theta)

        return doa, ranges

    def compute_doa_range(self, theta: np.ndarray):
        """ Given theta=[lambda, mu] return the angle and the range"""
        res = np.zeros(theta.shape)
        for idx, elem in enumerate(theta):
            lam, mu = elem
            alpha_lam = np.angle(np.log(lam))
            doa = np.arcsin(-alpha_lam / np.pi)
            alpha_mu = np.angle(np.log(mu))
            dist = 4 * alpha_mu / (np.pi * self.wavelength * np.cos(doa) ** 2)
            res[idx] = [doa, dist]

        return res[:, 0], res[:, 1]

    def find_global_theta(self, theta, values, num_sources):
        """Given 2(m-1) local minima, return num_sources global minima"""

        distances = np.abs(np.abs(values) - 1)
        sorted_idx = np.argsort(distances)
        opt_thet = theta[sorted_idx][:num_sources]

        return opt_thet

    def theta_prediction_loop(self, init_val):
        """

        :param alpha_lambda:
        :return:
        """
        alpha_mu = 0
        alpha_max = 0.25  # suggested value
        delta_alpha_mu = 0.01
        alpha_lambda = init_val
        q_values = [self.evalute_q_polynum(alpha_mu, alpha_lambda)]
        thetas, values = [], []
        while alpha_mu < alpha_max:
            alpha_lambda_new = self.step_prediction_alpha_lambda(alpha_lambda, delta_alpha_mu, alpha_mu)
            q_values.append(self.evalute_q_polynum(alpha_mu + delta_alpha_mu, alpha_lambda_new))
            if q_values[-1] >= 0 >= q_values[-2]:
                theta = np.array([alpha_lambda_new, alpha_mu + delta_alpha_mu]) + np.array([alpha_lambda, alpha_mu])
                theta, value = self.minimize_gen_mat_locally(0.5 * theta)
                # if (0.9 < np.abs(theta)).all() and (np.abs(theta) < 1.1).all():
                thetas.append(theta)
                values.append(value)
            alpha_mu += delta_alpha_mu
            alpha_lambda = alpha_lambda_new
        # the return value for thetas and values should be Nx2 and Nx1
        if len(thetas) == 0:
            return None, None
        thetas = np.concatenate(thetas).reshape(-1, 2)
        values = np.array(values).reshape(-1, 1)
        return thetas, values

    def minimize_gen_mat_locally(self, init_theta):  # TODO
        def F(params, W):
            m, _ = W.shape
            lambda_real = params[0]
            lambda_imag = params[1]
            mu_real = params[2]
            mu_imag = params[3]

            result_real = 0
            result_imag = 0

            for j in range(m):
                for k in range(m-1):
                    if j == k:
                        continue
                    term = W[j, k] * (lambda_real + 1j * lambda_imag) ** (k - j) * (
                                mu_real + 1j * mu_imag) ** (k ** 2 - j ** 2)
                    result_real += term.real
                    result_imag += term.imag
            return np.array([result_real, result_imag])

        def gradient(params, W):
            m, _ = W.shape
            lambda_real = params[0]
            lambda_imag = params[1]
            mu_real = params[2]
            mu_imag = params[3]

            grad_lambda_real = 0
            grad_lambda_imag = 0
            grad_mu_real = 0
            grad_mu_imag = 0

            for j in range(m):
                for k in range(m-1):
                    if j == k:
                        continue
                    term = W[j, k] * (lambda_real + 1j * lambda_imag) ** (k - j) * (
                            mu_real + 1j * mu_imag) ** (k ** 2 - j ** 2)
                    grad_lambda_real += (k - j) * term.real
                    grad_lambda_imag += (k - j) * term.imag
                    grad_mu_real += (k ** 2 - j ** 2) * term.real
                    grad_mu_imag += (k ** 2 - j ** 2) * term.imag

            return np.array([grad_lambda_real, grad_lambda_imag, grad_mu_real, grad_mu_imag])

        init_theta = np.array([init_theta[0].real, init_theta[0].imag, init_theta[1].real, init_theta[1].imag])
        result = minimize(lambda params: np.sum(F(params, self.gen_mat)), init_theta, jac=lambda params: gradient(params, self.gen_mat), method='L-BFGS-B')


        optimal_theta = result.x
        minimized_value_real, minimized_value_imag = F(optimal_theta, self.gen_mat)
        optimal_value = minimized_value_real + 1j * minimized_value_imag
        opt_lambda = optimal_theta[0] + 1j * optimal_theta[1]
        opt_mu = optimal_theta[2] + 1j * optimal_theta[3]
        optimal_theta = np.array([opt_lambda, opt_mu])

        return optimal_theta, optimal_value

    def step_prediction_alpha_lambda(self, alpha_lambda, delta_alpha_mu, alpha_mu):
        """
        This step envolve 2 steps: evaluation, using the Euler method, and correction, using the Newton method.
        :param alpha_lambda:
        :param delta_alpha_mu:
        :param alpha_mu:
        :return:
        """
        # Euler's method: using numerical differential
        eps = 10 ** -6
        diff_mu = (self.evalute_p_polynum(alpha_lambda, alpha_mu + eps)
                   - self.evalute_p_polynum(alpha_lambda, alpha_mu)) / eps
        diff_lambda = (self.evalute_p_polynum(alpha_lambda + eps, alpha_mu)
                       - self.evalute_p_polynum(alpha_lambda, alpha_mu)) / eps
        diff = - diff_mu / diff_lambda
        alpha_lambda_new = alpha_lambda + diff * delta_alpha_mu

        # 2 steps of Newton's method
        diff_lambda = (self.evalute_p_polynum(alpha_lambda_new + eps, alpha_mu)
                       - self.evalute_p_polynum(alpha_lambda_new, alpha_mu)) / eps
        alpha_lambda_new -= (1 / diff_lambda) * self.evalute_p_polynum(alpha_lambda_new, alpha_mu)

        return alpha_lambda_new

    def calc_p_polynum_coeff(self) -> np.ndarray:
        """
        The coefficeints are ensamble of the sum of a diagonal in the kernel matrix, multiply by the index.
        :param kernel:
        :return:
        """
        coeff = []
        diag_idx = np.linspace(-self.gen_mat.shape[0] + 1,
                               self.gen_mat.shape[0] + 1,
                               2 ** self.gen_mat.shape[0] - 1,
                               endpoint=False, dtype=int)
        for idx in diag_idx:
            tmp = np.sum(self.gen_mat.diagonal(idx))
            if idx >= 0:
                tmp *= idx
            else:
                tmp *= (self.gen_mat.shape[0] + idx)
            coeff.append(tmp)

        return np.array(coeff)

    def find_roots(self, coeff: np.ndarray) -> np.ndarray:
        """
        this function is taking the coefficients and return the roots of the polynomial.
        :param coeff:
        :return:
        """
        A = np.diag(np.ones((len(coeff) - 2,), coeff.dtype), -1)
        if np.abs(coeff[0]) == 0:
            A[0, :] = -coeff[1:] / (coeff[0] + 10 ** (-9))
        else:
            A[0, :] = -coeff[1:] / coeff[0]
        roots = np.linalg.eigvals(A)

        return roots

    def evalute_p_polynum(self, alpha_lambda: float, alpha_mu: float) -> complex:

        mu = np.exp(1j * alpha_mu)
        lamb = np.exp(1j * alpha_lambda)
        res = 0
        for j in range(self.gen_mat.shape[0]):
            for k in range(self.gen_mat.shape[1]):
                res += (k - j) * self.gen_mat[j, k] * lamb ** (k - j) ** mu ** (k ** 2 - j ** 2)

        return 1j * res

    def evalute_q_polynum(self, alpha_mu: float, alpha_lambda: float | np.ndarray) -> np.ndarray | complex:
        """

        :param alpha_lambda:
        :param alpha_mu:
        :return:
        """
        mu = np.exp(1j * alpha_mu)
        if type(alpha_lambda) != np.ndarray:
            alpha_lambda = [alpha_lambda]
        res = np.zeros(len(alpha_lambda),dtype=complex)
        for j in range(self.gen_mat.shape[0]):
            for k in range(self.gen_mat.shape[1]):
                for idx, alpha in enumerate(alpha_lambda):
                    lamb = np.exp(1j * alpha)
                    res[idx] += (k ** 2 - j ** 2) * self.gen_mat[j, k] * lamb ** (k - j) * mu ** (k ** 2 - j ** 2)

        return 1j * res


if __name__ == '__main__':
    pass
