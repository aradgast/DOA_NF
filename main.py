from src.methods import MUSIC, MUSIC2D
from src.signal import Signal
from utils.functions import plot_angles_on_unit_circle
from utils.simulation import MTSimulation
from src.loss import compute_mse_loss, compute_rmpse_loss
import numpy as np
if __name__ == '__main__':
    pass
    # T = [100]
    # S = [2, 3, 4]
    # M = 8
    # wavelength = 1
    # array_geometry = 'ULA'
    # snr = [0, 5, 10, 15]
    # method = MUSIC(array_geometry, M, wavelength, S[0])
    # signal = Signal(wavelength, array_geometry, num_sources=S[0], num_sensors=M)
    # # sample = signal.generate(snr[0], np.deg2rad([-50, 60]), 50, S, M)
    # # pred = method.compute_predictions(sample)
    # # print(np.rad2deg(pred))
    # sim = MTSimulation(iteration_num=10,
    #                    method=method,
    #                    signal=signal,
    #                    loss=compute_mse_loss,
    #                    snr_range=snr,
    #                    source_range=S,
    #                    sample_range=T)
    # results = sim.run_snr_sources()
    method = MUSIC2D(array_geometry='ULA',
                     num_sensors=8,
                     wavelength=1,
                     num_sources=1)
    signal = Signal(array_geometry='ULA',
                    num_sensors=8,
                    wavelength=1,
                    num_sources=1)
    sample = signal.generate(snr=30,
                             angles=np.deg2rad([35]),
                             distances=[10],
                             num_samples=100)
    pred = method.compute_predictions(sample)
    print(f"Angles: {np.rad2deg(pred[0])}")
    print(f"Radius: {pred[1]}")



