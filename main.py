from src.methods import MUSIC, MUSIC2D
from src.signal import Signal
from utils.functions import plot_angles_on_unit_circle
from utils.simulation import MTSimulation
from src.loss import compute_mse_loss, compute_rmpse_loss
import numpy as np
# from utils.functions import choose_angles, choose_distances, calculate_fraunhofer_distance
from src.modules import Module
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
    # snr = [5, 10, 15, 20, 25]
    snr = [5, 10, 15, 20]
    T = [25]
    # T = [100]
    S = [1]
    M = 5
    wavelength = 1
    array_geometry = 'ULA'
    module = Module(array_geometry=array_geometry, num_sensors=M, wavelength=wavelength, is_2d=True)

    method = MUSIC2D(module=module,
                     num_sources=S[0])
    signal = Signal(module=module,
                    num_sources=S[0])
    print(f"Fraunhofer distance: {module.calculate_fraunhofer_distance()}")
    ################## SINGLE RUN ##################
    # angles = module.choose_angles(S[0])
    # distances = module.choose_distances(S[0])
    # angles = [-np.pi/3]
    # distances = [12.5]
    # print(f"True Angles: {np.rad2deg(angles)}")
    # print(f"True Distances: {distances}")
    # sample = signal.generate_2d(snr=snr[-1],
    #                             angles=angles,
    #                             distances=distances,
    #                             num_samples=T[-1])
    # pred_angles, pred_distances = method.compute_predictions(sample)
    # print(f"Angles: {np.rad2deg(pred_angles)}")
    # print(f"Radius: {pred_distances}")
    ###########################################################
    ################## MONTE CARLO ############################
    sim = MTSimulation(iteration_num=100,
                       module=module,
                       method=method,
                       signal=signal,
                       loss=compute_mse_loss,
                       snr_range=snr,
                       source_range=S,
                       sample_range=T,
                       is_2d=True)
    # results = sim.run_snr_sources(show_plot=False, save_plot=True)
    results = sim.run_snr_samples(show_plot=True, save_plot=True)
    # # results = sim.run_NumberofSnapshot(show_plot=False, save_plot=True)
    # sim.run_rmse_distance(15, save_plot=True, show_plot=True)
    ###########################################################
