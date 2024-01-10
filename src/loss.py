import numpy as np
import scipy as sc
from sympy.utilities.iterables import multiset_permutations


def compute_mse_loss(predictions:list, doa:list):
    """
    Compute the MSE between the predictions and the true DOAs
    :param predictions: list
    :param doa: list
    :return: float
    """
    doa = np.array(sorted(doa))
    predictions = np.array(sorted(predictions))
    return np.mean((np.array(predictions) - np.array(doa)) ** 2)

def compute_rmpse_loss(predictions_doa:list, doa:list, predictions_distance: list, distance: list):
    """
    Compute the RMPSE between the predictions and the true DOAs
    :param predictions: list
    :param doa: list
    :return: float
    """
    doa = np.array(doa)
    predictions_perm_doa = multiset_permutations(predictions_doa)
    predictions_perm_dist = multiset_permutations(predictions_distance)
    rmspe_list = []
    for pred_doa, pred_dist in zip(predictions_perm_doa, predictions_perm_dist):
        # Calculate error with modulo pi
        pred_doa = np.array(pred_doa)
        pred_dist = np.array(pred_dist)
        error = (((pred_doa - doa) + (np.pi / 2)) % np.pi) - np.pi / 2
        err_dist = (pred_dist - distance) * np.pi / 10
        error += err_dist
        # Calculate RMSE over all permutations
        rmspe_val = (1 / np.sqrt(len(doa))) * np.linalg.norm(error)
        rmspe_list.append(rmspe_val)
    rmspe_min = np.min(rmspe_list)
    return rmspe_min
