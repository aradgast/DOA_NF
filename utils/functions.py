import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import random


def compute_covariance_matrix(signal):
    """
    Compute the covariance matrix of the signal
    :param signal: 2D array of shape (num_samples, num_channels)
    :return: 2D array of shape (num_channels, num_channels)
    """
    return np.cov(signal)




def plot_angles_on_unit_circle(true, predections):
    # plot the angles theats and angles predictions on the half unit circle: -90 to 90 degrees
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.deg2rad(true), np.ones(len(true)), 'ro', label='true')
    ax.plot(np.deg2rad(predections), np.ones(len(predections)), 'bo', label='predicted')
    ax.set_title("Angles on unit circle", va='bottom')
    # ax.set_xticks(np.pi / 180. * np.linspace(-90, 90, 18, endpoint=False))
    # ax.set_xticklabels(np.linspace(-90, 90, 18, endpoint=False))
    ax.legend()
    plt.show()


def set_unified_seed(seed:int = 42):
    random.seed(seed)
    np.random.seed(seed)
