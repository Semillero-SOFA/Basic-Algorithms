'''
    Author: Kevin Martinez
    Date: 19-03-2024
    Description: This file contains functions commonly used in the SOFA Semin.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

mod_dict = {
    0: -3 + 3j, 1: -3 + 1j, 2: -3 - 3j, 3: -3 - 1j,
    4: -1 + 3j, 5: -1 + 1j, 6: -1 - 3j, 7: -1 - 1j,
    8:  3 + 3j, 9:  3 + 1j, 10: 3 - 3j, 11: 3 - 1j,
    12: 1 + 3j, 13: 1 + 1j, 14: 1 - 3j, 15: 1 - 1j
}

demod_dict = {
    -3 + 3j: 0, -3 + 1j: 1, -3 - 3j: 2, -3 - 1j: 3,
    -1 + 3j: 4, -1 + 1j: 5, -1 - 3j: 6, -1 - 1j: 7,
     3 + 3j: 8,  3 + 1j: 9,  3 - 3j: 10, 3 - 1j: 11,
     1 + 3j: 12, 1 + 1j: 13, 1 - 3j: 14, 1 - 1j: 15

}

def add_noise(signal, noise_db):
    """ Add noise to a signal at a given dB level.

    Args:
        signal (np.array): The signal to add noise to.
        noise_dB (float): The noise level in dB.
    """
    
    X_avg_p = np.mean(signal ** 2)
    X_avg_db = 10 * np.log10(X_avg_p)
    noise_avg_db_r = X_avg_db - noise_db
    noise_avg_p_r = 10 ** (noise_avg_db_r / 10)
    mean_noise = 0
    noise_r = np.random.normal(mean_noise, np.sqrt(noise_avg_p_r), signal.shape)
    noisy_signal = signal + noise_r
    return noisy_signal

def create_16qam_const(Ns=10000, noise='none', noise_db=0):
    """ Create a 16-QAM constellation.

    Args:
        Ns (int, optional): The number of symbols. Defaults to 10000.
        noise (str, optional): The noise type ['none', 'awgn']. Defaults to 'none'.
        noise_db (int, optional): The noise level in dB. Defaults to 0.
    """
    
    y = np.random.randint(16, size=Ns)
    syms = np.array([mod_dict[x] for x in y])
    Xr = np.real(syms)
    Xi = np.imag(syms)
    if noise == 'none':
        X = np.vstack((Xr, Xi)).T
        return X, y
    elif noise == 'awgn':
        Xr_ch = add_noise(Xr, noise_db)
        Xi_ch = add_noise(Xi, noise_db)
        X = np.vstack((Xr_ch, Xi_ch)).T
        return X, y
    else:
        print('Invalid noise type.')
        return None
    
def plot_16qam(X, y, grid=False, lim=[-4, 4]):
    """ Plot a 16-QAM constellation.

    Args:
        X (np.array): The 16-QAM constellation.
        y (np.array): The labels of the constellation.
        grid (bool, optional): Whether to show the grid. Defaults to False.
        lim (list, optional): The limits of the plot. Defaults to [-4, 4].
    """
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20, edgecolors='k', zorder=2)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel('In-Phase', fontsize=14)
    plt.ylabel('Quadrature', fontsize=14)
    plt.xticks(np.arange(lim[0], lim[1] + 1, 1), fontsize=12)
    plt.yticks(np.arange(lim[0], lim[1] + 1, 1), fontsize=12)
    if grid: plt.grid(linestyle='--', zorder=0)
    plt.show()

def modulate_16qam(signal):
    """ Modulate a signal using 16-QAM format.

    Args:
        signal (np.array): The signal to modulate.
    """
    
    mod_signal = np.zeros(len(signal), dtype=complex)
    for i in range(len(signal)):
        mod_signal[i] = mod_dict[signal[i]]
    return mod_signal

def demodulate_16qam(signal):
    """ Demodulate a signal using 16-QAM format.

    Args:
        signal (np.array): The signal to demodulate.
    """
    
    demod_signal = np.zeros(len(signal), dtype=int)
    for i in range(len(signal)):
        demod_signal[i] = demod_dict[signal[i]]
    return demod_signal

def bit_error_rate(y_true, y_pred):
    """ Calculate the bit error rate (BER) of two signals.

    Args:
        y_true (np.array): The true signal.
        y_pred (np.array): The predicted signal.
    """
    
    true = ''.join([format(x, '04b') for x in y_true])
    pred = ''.join([format(x, '04b') for x in y_pred])
    ber = np.sum([1 for i in range(len(true)) if true[i] != pred[i]]) / len(true)
    return ber

def symbol_error_rate(y_true, y_pred):
    """ Calculate the symbol error rate (SER) of two signals.

    Args:
        y_true (np.array): The true signal.
        y_pred (np.array): The predicted signal.
    """
    
    ser = np.sum([1 for i in range(len(y_true)) if y_true[i] != y_pred[i]]) / len(y_true)
    return ser

def sync_signals(tx_signal, rx_signal):
    """ Synchronize two signals.

    Args:
        tx_signal (np.array): The transmitted signal.
        rx_signal (np.array): The received signal.
    """
    
    tx_s = np.concatenate((tx_signal, tx_signal))
    corr = np.abs(np.correlate(np.abs(tx_s) - np.mean(np.abs(tx_s)), 
                               np.abs(rx_signal) - np.mean(np.abs(rx_signal)), 
                               mode='full'))
    delay = np.argmax(corr) - len(rx_signal) + 1
    sync_signal = tx_s[delay:]
    sync_signal = sync_signal[:len(rx_signal)]
    return sync_signal

def fit_curve(step, x, y, deg=2):
    """ Fit a curve to a set of points.

    Args:
        step (np.array): The step values of the curve.
        x (np.array): The x-axis values.
        y (np.array): The y-axis values.
        deg (int, optional): The degree of the polynomial. Defaults to 2.
    """
    
    p = np.polyfit(x, y, deg)
    z = np.poly1d(p)
    curve = z(step)
    return curve

def demapper_sym(symbols_I, symbols_Q, threshold=2.0):
    """ Demap complex values to symbols.

    Args:
        symbols_I (np.array): The In-Phase symbols.
        symbols_Q (np.array): The Quadrature symbols.
        threshold (float, optional): The threshold value. Defaults to 2.0.
    """
    
    symbols = np.zeros(len(symbols_I), dtype=int)
    for i in range(len(symbols_I)):
        if symbols_I[i] <= -threshold and symbols_Q[i] >= threshold: # -3 + 3j
            symbols[i] = 0
        elif symbols_I[i] <= -threshold and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold: # -3 + 1j
            symbols[i] = 1
        elif symbols_I[i] <= -threshold and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold: # -3 - 1j
            symbols[i] = 3
        elif symbols_I[i] <= -threshold and symbols_Q[i] <= -threshold: # -3 - 3j
            symbols[i] = 2
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] >= threshold: # -1 + 3j
            symbols[i] = 4
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold: # -1 + 1j
            symbols[i] = 5
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold: # -1 - 1j
            symbols[i] = 7
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] <= -threshold: # -1 - 3j
            symbols[i] = 6
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] >= threshold: # 1 + 3j
            symbols[i] = 12
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold: # 1 + 1j
            symbols[i] = 13
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold: # 1 - 1j
            symbols[i] = 15
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] <= -threshold: # 1 - 3j
            symbols[i] = 14
        elif symbols_I[i] >= threshold and symbols_Q[i] >= threshold: # 3 + 3j
            symbols[i] = 8
        elif symbols_I[i] >= threshold and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold: # 3 + 1j
            symbols[i] = 9
        elif symbols_I[i] >= threshold and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold: # 3 - 1j
            symbols[i] = 11
        elif symbols_I[i] >= threshold and symbols_Q[i] <= -threshold: # 3 - 3j
            symbols[i] = 10
        else:
            symbols[i] = -1
    return symbols

def remaining_labels_kdt(labels, X, predict_mode='nearest', radius=1.0, k=3):
    """ Get remaining labels using a KD-Tree structure.

    Args:
        labels (np.array): The predicted labels.
        X (np.array): The data to train the KD-Tree.
        predict_mode (str, optional): The prediction mode ['nearest', 'radius']. Defaults to 'nearest'.
        radius (float, optional): The radius of the KD-Tree. Defaults to 1.0.
        k (int, optional): The number of neighbors to use. Defaults to 3.
    """
    
    new_labels = np.copy(labels)
    noise = X[new_labels == -1]
    noise_idx = np.where(new_labels == -1)[0]
    non_noise = X[new_labels != -1]
    non_noise_idx = np.where(new_labels != -1)[0]
    if len(noise) == 0:
        return new_labels
    kdt = KDTree(non_noise)
    if predict_mode == 'nearest':
        _, ind = kdt.query(noise, k=k)
    elif predict_mode == 'radius':
        ind = kdt.query_radius(noise, r=radius)
    else:
        print('Invalid prediction mode.')
        return None
    for i in range(len(noise)):
        neighbors_idx = new_labels[non_noise_idx[ind[i]]]
        label = np.argmax(np.bincount(neighbors_idx))
        new_labels[noise_idx[i]] = label
    return new_labels
