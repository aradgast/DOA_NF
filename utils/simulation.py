import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from utils.functions import *
from src.modules import Module


class MTSimulation:
    def __init__(self, module: Module, iteration_num: int, method, signal, loss, snr_range: list,
                 source_range: list, sample_range: list, is_2d: bool = False):
        self.module = module
        self.iteration_num = iteration_num
        self.method = method
        self.signal = signal
        self.snr_range = snr_range
        self.source_range = source_range
        self.sample_range = sample_range
        self.loss = loss
        self.is_2d = is_2d

    def run_snr_samples(self, show_plot: bool = True, save_plot: bool = False):
        if self.is_2d:
            return self.__run_snr_samples_2D(show_plot, save_plot)
        else:
            return self.__run_snr_samples_1D(show_plot, save_plot)

    def run_snr_sources(self, show_plot: bool = True, save_plot: bool = False):
        if self.is_2d:
            return self.__run_snr_sources_2D(show_plot, save_plot)
        else:
            return self.__run_snr_sources_1D(show_plot, save_plot)

    def run_NumberofSnapshot(self, show_plot: bool = True, save_plot: bool = False):
        if self.is_2d:
            return self.__run_NumberofSnapshot_2D(show_plot, save_plot)
        else:
            return self.__run_NumberofSnapshot_1D(show_plot, save_plot)

    def __run_snr_samples_1D(self, show_plot: bool = True, save_plot: bool = False):
        """
        Run the simulation
        :return: None
        """
        # Initialize the results array
        results = np.zeros((len(self.snr_range), len(self.sample_range)))
        # Run the simulation
        S = self.source_range[0]
        doa = self.module.choose_angles(S)
        for snr_idx, snr in enumerate(self.snr_range):
            for t_idx, t in enumerate(self.sample_range):
                for i in range(self.iteration_num):
                    # Generate the signal
                    samples = self.signal.generate(snr=snr, angles=doa, num_samples=t)
                    # Compute the predictions
                    predictions = self.method.compute_predictions(samples)
                    # Compute the loss
                    loss_i = self.loss(predictions, doa)
                    # Store the results
                    results[snr_idx, t_idx] += (loss_i / float(self.iteration_num))
                print(f'SNR = {snr}, T = {t}:  MSE = {results[snr_idx, t_idx]}')
        # Plot the results
        plt.figure()
        plt.title(f'MSE vs SNR and number of samples, S = {S}, DOA = {np.rad2deg(doa)}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('MSE (dB)')
        for idx, T in enumerate(self.sample_range):
            plt.plot(self.snr_range, 10 * np.log10(results[:, idx]), label=f'T = {T}')

        plt.legend()
        plt.grid()
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_1D\run_snr_samples.jpeg")
        return results

    def __run_snr_samples_2D(self, show_plot: bool = True, save_plot: bool = False):
        """
                Run the simulation
                :return: None
                """
        # Initialize the results array
        results_angles = np.zeros((len(self.snr_range), len(self.sample_range)))
        results_distances = np.zeros((len(self.snr_range), len(self.sample_range)))
        # Run the simulation
        S = self.source_range[0]
        doa = self.module.choose_angles(S)
        doa = np.deg2rad([45, 60])
        # dist = self.module.choose_distances(S)
        dist = [15, 30]
        for snr_idx, snr in enumerate(self.snr_range):
            for t_idx, t in enumerate(self.sample_range):
                loss_angles = []
                loss_dist = []
                for i in range(self.iteration_num):
                    # Generate the signal
                    samples = self.signal.generate_2d(snr=snr, angles=doa, distances=dist,
                                                      num_samples=t, num_sources=S)
                    # Compute the predictions
                    predictions_angles, predictions_dist = self.method.compute_predictions(samples)
                    # Compute the loss
                    loss_angles.append(np.array(doa)-np.sort(predictions_angles))
                    loss_dist.append(np.array(dist)-np.sort(predictions_dist))
                # Store the results
                results_angles[snr_idx, t_idx] = np.sqrt(np.mean(np.power(loss_angles, 2)))
                results_distances[snr_idx, t_idx] = np.sqrt(np.mean(np.power(loss_dist, 2)))
                print(f'SNR = {snr}, T = {t}:  '
                      f'RMSE(angles, dist) = ({results_angles[snr_idx, t_idx]}, {results_distances[snr_idx, t_idx]})')
        # Plot the results
        plt.subplot(1, 2, 1)
        plt.title(f'DOA = {np.rad2deg(doa)}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('RMSE(angle)')
        for idx, T in enumerate(self.sample_range):
            plt.plot(self.snr_range, results_angles[:, idx], label=f'T = {T}')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title(f'Distances = {dist}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('RMSE(distance)')
        for idx, T in enumerate(self.sample_range):
            plt.plot(self.snr_range, results_distances[:, idx], label=f'T = {T}')
        plt.legend()
        plt.grid()

        plt.suptitle(f"RMSE vs SNR and number of samples, S = {S}", fontsize=16)
        plt.tight_layout()

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_2D\run_snr_samples.jpeg")
        return results_angles, results_distances

    def __run_snr_sources_1D(self, show_plot: bool = True, save_plot: bool = False):
        """
        Run the simulation
        :return: None
        """
        # Initialize the results array
        results = np.zeros((len(self.snr_range), len(self.source_range)))
        # Run the simulation
        T = self.sample_range[0]
        for s_idx, s in enumerate(self.source_range):
            doa = self.module.choose_angles(s)
            print(f"DOA = {np.rad2deg(doa)}")
            for snr_idx, snr in enumerate(self.snr_range):
                for i in range(self.iteration_num):
                    # Generate the signal
                    samples = self.signal.generate(snr=snr, angles=doa, num_samples=T, num_sources=s)
                    # Compute the predictions
                    predictions = self.method.compute_predictions(samples, num_sources=s)
                    # Compute the loss

                    loss_i = self.loss(predictions, doa)
                    # Store the results
                    results[snr_idx, s_idx] += (loss_i / float(self.iteration_num))
                print(f'SNR = {snr}, S = {s}:  MSE = {results[snr_idx, s_idx]}')
        # Plot the results
        plt.figure()
        plt.title(f'MSE vs SNR and number of sources, T = {T}')
        plt.xlabel('SNR (dB)')
        plt.ylabel('MSE (dB)')
        for idx, S in enumerate(self.source_range):
            plt.plot(self.snr_range, 10 * np.log10(results[:, idx]), label=f'S = {S}')

        plt.legend()
        plt.grid()
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_1D\run_snr_sources.jpeg")

        return results

    def __run_snr_sources_2D(self, show_plot: bool = True, save_plot: bool = False):
        """

        :param show_plot:
        :param save_plot:
        :return:
        """
        # Initialize the results array
        results_angles = np.zeros((len(self.snr_range), len(self.source_range)))
        results_dist = np.zeros((len(self.snr_range), len(self.source_range)))
        # Run the simulation
        T = self.sample_range[0]
        for s_idx, s in enumerate(self.source_range):
            doa = self.module.choose_angles(s)
            dist = self.module.choose_distances(s)
            print(f"DOA = {np.rad2deg(doa)}")
            for snr_idx, snr in enumerate(self.snr_range):
                for i in range(self.iteration_num):
                    # Generate the signal
                    samples = self.signal.generate_2d(snr=snr, angles=doa, distances=dist, num_samples=T, num_sources=s)
                    # Compute the predictions
                    predictions_angles, predictions_dist = self.method.compute_predictions(samples, num_sources=s)
                    # Compute the loss

                    loss_i_angles = self.loss(predictions_angles, doa)
                    loss_i_dist = self.loss(predictions_dist, dist)
                    # Store the results
                    results_angles[snr_idx, s_idx] += (loss_i_angles / float(self.iteration_num))
                    results_dist[snr_idx, s_idx] += (loss_i_dist / float(self.iteration_num))
                print(f'SNR = {snr}, S = {s}:  MSE(angles, distances) = '
                      f'({results_angles[snr_idx, s_idx]}, {results_dist[snr_idx, s_idx]})')
        # Plot the results
        plt.subplot(1, 2, 1)
        plt.title(f'DOA')
        plt.xlabel('SNR (dB)')
        plt.ylabel('MSE(Angle) (dB)')
        for idx, S in enumerate(self.source_range):
            plt.plot(self.snr_range, 10 * np.log10(results_angles[:, idx]), label=f'S = {S}')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title(f'Distances')
        plt.xlabel('SNR (dB)')
        plt.ylabel('MSE(distance) (dB)')
        for idx, S in enumerate(self.source_range):
            plt.plot(self.snr_range, 10 * np.log10(results_dist[:, idx]), label=f'S = {S}')
        plt.legend()
        plt.grid()

        plt.suptitle(f'MSE vs SNR and number of sources, T = {T}', fontsize=16)
        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_2D\run_snr_sources.jpeg")

        return results_angles

    def __run_NumberofSnapshot_1D(self, show_plot: bool = True, save_plot: bool = False):
        """
        :param show_plot:
        :param save_plot:
        :return:
        """
        # Initialize the results array
        results = np.zeros(len(self.sample_range))
        # Run the simulation
        SNR = self.snr_range[0]
        self.source_range = self.source_range[0]
        doa = self.module.choose_angles(self.source_range)
        print(f"")
        for s_idx, num_samples in enumerate(self.sample_range):
            for i in range(self.iteration_num):
                # Generate the signal
                samples = self.signal.generate(snr=SNR, angles=doa, num_samples=num_samples,
                                               num_sources=self.source_range)
                # Compute the predictions
                predictions = self.method.compute_predictions(samples, num_sources=self.source_range)
                # Compute the loss

                loss_i = self.loss(predictions, doa)
                # Store the results
                results[s_idx] += (loss_i / float(self.iteration_num))
            print(f'Number of Snapshots = {num_samples}:  MSE = {results[s_idx]}')
        # Plot the results
        plt.figure()
        plt.title(f'MSE vs NumberofSnapshot, SNR = {SNR}, S = {self.source_range}, DOA = {np.rad2deg(doa)}')
        plt.xlabel('Number of Snapshot')
        plt.ylabel('MSE (dB)')
        plt.semilogx(self.sample_range, 10 * np.log10(results))
        plt.grid()
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_1D\run_NumberofSnapshot.jpeg")

        return results

    def __run_NumberofSnapshot_2D(self, show_plot: bool = True, save_plot: bool = False):
        """
                :param show_plot:
                :param save_plot:
                :return:
                """
        # Initialize the results array
        results_angles = np.zeros(len(self.sample_range))
        results_dist = np.zeros(len(self.sample_range))
        # Run the simulation
        SNR = self.snr_range[0]
        S = self.source_range[0]
        # doa = self.module.choose_angles(S)
        doa = np.deg2rad([45, 60])
        # dist = self.module.choose_distances(S)
        dist = [15, 30]
        for t_idx, snapshots in enumerate(self.sample_range):
            loss_angles = []
            loss_dist = []
            for i in range(self.iteration_num):
                # Generate the signal
                samples = self.signal.generate_2d(snr=SNR, angles=doa, distances=dist,
                                                  num_samples=snapshots, num_sources=S)
                # Compute the predictions
                predictions_angles, predictions_dist = self.method.compute_predictions(samples, num_sources=S)
                # Compute the loss
                loss_angles.append(np.array(doa) - np.sort(predictions_angles))
                loss_dist.append(np.array(dist) - np.sort(predictions_dist))
                # Store the results
            results_angles[t_idx] += np.sqrt(np.mean(np.power(loss_angles, 2)))
            results_dist[t_idx] += np.sqrt(np.mean(np.power(loss_dist, 2)))
            print(f'Number of Snapshots = {snapshots}:  '
                  f'RMSE(angles, distances) = ({results_angles[t_idx]}, {results_dist[t_idx]})')
        # Plot the results
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title(f'DOA = {np.rad2deg(doa)}')
        plt.xlabel('Number of Snapshot')
        plt.ylabel('RMSE(angle)')
        plt.semilogx(self.sample_range, results_angles)
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title(f'Distance = {dist}')
        plt.xlabel('Number of Snapshot')
        plt.ylabel('RMSE(dist)')
        plt.semilogx(self.sample_range, results_dist)
        plt.grid()

        plt.suptitle(f'RMSE vs NumberofSnapshot, SNR = {SNR}, S = {S}')
        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_2D\run_NumberofSnapshot.jpeg")

        return results_angles, results_dist
