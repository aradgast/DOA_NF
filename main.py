from src.methods import MUSIC, MUSIC2D
from src.signal import Signal
from utils.functions import plot_angles_on_unit_circle
from utils.simulation import MTSimulation
from src.loss import compute_mse_loss, compute_rmpse_loss
import numpy as np
from utils.functions import choose_angles, choose_distances, calculate_fraunhofer_distance

if __name__ == '__main__':
    pass
    ############### MUSIC ########################

    # T = [10, 50, 100, 200, 500, 1000]
    # S = [2, 3, 4]
    # M = 8
    # wavelength = 1
    # array_geometry = 'ULA'
    # snr = [5, 10, 15, 20, 25, 30]
    # method = MUSIC(array_geometry, M, wavelength, S[0])
    # signal = Signal(wavelength, array_geometry, num_sources=S[0], num_sensors=M)
    ################## SINGLE RUN ##################
    # sample = signal.generate(snr[0], np.deg2rad([-50, 60]), 50, S, M)
    # pred = method.compute_predictions(sample)
    # print(np.rad2deg(pred))
    ################################################

    ################## MONTE CARLO #################
    # sim = MTSimulation(iteration_num=100,
    #                    method=method,
    #                    signal=signal,
    #                    loss=compute_mse_loss,
    #                    snr_range=snr,
    #                    source_range=S,
    #                    sample_range=T)
    # results = sim.run_snr_sources()
    # results = sim.run_snr_samples()
    # results = sim.run_NumberofSnapshot()
    #################################################

    ############### MUSIC 2D ########################
    S = 2
    M = 7
    wavelength = 1
    array_geometry = 'ULA'
    angles = choose_angles(1, min_gap=2, max_gap=180)
    distances = choose_distances(S, array_geometry, M, wavelength, min_gap=1, max_gap=100)
    print(f"True Angles: {np.sort(np.rad2deg(angles))}")
    print(f"True Distances: {np.sort(distances)},"
          f" Fraunhofer distance: {calculate_fraunhofer_distance(array_geometry, M, wavelength)[0]}")
    method = MUSIC2D(array_geometry=array_geometry,
                     num_sensors=M,
                     wavelength=wavelength,
                     num_sources=S)
    signal = Signal(array_geometry=array_geometry,
                    num_sensors=M,
                    wavelength=wavelength,
                    num_sources=S)
    # for snr in [10, 20, 30, 40, 50]:
    #     for T in [10, 50, 100, 200]:
    # print(f"###### SNR = {snr}, T = {T} ########")
    snr = [0, 5, 10, 15, 25, 30]
    T = [10, 50, 100, 200, 500, 1000]
    sample = signal.generate_2d(snr=snr[0],
                                angles=angles,
                                distances=distances,
                                num_samples=T[0])
    pred_angles, pred_distances = method.compute_predictions(sample)
    print(f"Angles: {np.sort(np.rad2deg(pred_angles))}")
    print(f"Radius: {np.sort(pred_distances)}")
    ###########################################################
    S = [2, 3, 4]
    sim = MTSimulation(iteration_num=100,
                       method=method,
                       signal=signal,
                       loss=compute_mse_loss,
                       snr_range=snr,
                       source_range=S,
                       sample_range=T,
                       is_2d=True)
    results = sim.run_snr_sources()
    results = sim.run_snr_samples()
    results = sim.run_NumberofSnapshot()
