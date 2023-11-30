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

def compute_rmpse_loss(predictions:list, doa:list):
    """
    Compute the RMPSE between the predictions and the true DOAs
    :param predictions: list
    :param doa: list
    :return: float
    """
    doa = np.array(doa)
    predictions_perm = multiset_permutations(predictions)
    rmspe_list = []
    for prediction in predictions_perm:
        # Calculate error with modulo pi
        prediction = np.array(prediction)
        error = (((prediction - doa) + (np.pi / 2)) % np.pi) - np.pi / 2
        # Calculate RMSE over all permutations
        rmspe_val = (1 / np.sqrt(len(doa))) * np.linalg.norm(error)
        rmspe_list.append(rmspe_val)
    rmspe_min = np.min(rmspe_list)
    return rmspe_min
