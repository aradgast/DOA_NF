import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from utils.functions import *


class MTSimulation:
    def __init__(self, iteration_num: int, method, signal, loss, snr_range: list,
                 source_range: list, sample_range: list):
        self.iteration_num = iteration_num
        self.method = method
        self.signal = signal
        self.snr_range = snr_range
        self.source_range = source_range
        self.sample_range = sample_range
        self.loss = loss

    def run_snr_samples(self):
        """
        Run the simulation
        :return: None
        """
        # Initialize the results array
        results = np.zeros((len(self.snr_range), len(self.sample_range)))
        # Run the simulation
        S = self.source_range[0]
        doa = choose_angles(S)
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
        # plt.show()
        plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_1D\run_snr_samples.jpeg")
        return results

    def run_snr_sources(self):
        """
                Run the simulation
                :return: None
                """
        # Initialize the results array
        results = np.zeros((len(self.snr_range), len(self.source_range)))
        # Run the simulation
        T = self.sample_range[0]
        for s_idx, s in enumerate(self.source_range):
            doa = choose_angles(s, max_gap=10)
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
        # plt.show()
        plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_1D\run_snr_sources.jpeg")

        return results

    def run_NumberofSnapshot(self):
        """

        :return:
        """
        # Initialize the results array
        results = np.zeros(len(self.sample_range))
        # Run the simulation
        SNR = self.snr_range[0]
        self.source_range = self.source_range[0]
        doa = choose_angles(self.source_range)
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
        # plt.show()
        plt.savefig(r"C:\Users\agast\Documents\University\DOA_NF\Results\MUSIC_1D\run_NumberofSnapshot.jpeg")
        return results
