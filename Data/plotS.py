import numpy as np
import matplotlib.pyplot as plt
import os
N = 100
delta = 0.01
os.chdir('uniruns')
# Path = 'LasingRungeEntropy.npy'
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def name(path):
    if 'Above' in path:
        return 'above the masing threshold'
    elif 'Lasing' in path:
        return 'at the masing threshold'
    elif 'Below' in path:
        return 'below the masing threshold'
    else:
        raise TypeError('Unknown')


def name2(path):
    return path.replace('.npy', '.pdf')


def plot(path, val):
    data = parser(path)
    xarray = np.array([i for i in range(len(data))]) * delta
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(xarray, data,'-', markersize=1)
    ax.grid()
    ax.set_xlabel(r'Time, $\gamma_h^{-1}$', size=15)
    ax.set_ylabel(r'Entropy $k_b$', size=15)
    ax.set_title(f'Entropy of the maser, when \noperating {name(path)}.', size=17, pad=2)
    ax.annotate(text=val, xy=[0, 0.9 * np.amax(data)], size=16)
    plt.tight_layout()
    #plt.show()
    plt.savefig(name2(path))


def plot2(path):
    data = parser(path)
    xarray = np.array([i for i in range(len(data))]) * delta
    fig, ax = plt.subplots(2, 1, figsize=(7, 5))
    ax[0].plot(xarray, data, '-', markersize=1)
    ax[0].grid()
    ax[0].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15)
    ax[0].set_ylabel(r'Entropy, $k_b$', size=15)
    ax[0].set_title(f'Entropy of the maser, when \noperating {name(path)}.', size=17, pad=2)
    ax[0].annotate(text='(a)', xy=[1, 0.85 * np.amax(data)], size=16)
    ax[1].plot(xarray, data, '-', markersize=1)
    ax[1].grid()
    ax[1].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15)
    ax[1].set_ylabel(r'Entropy, $k_b$', size=15)
    ax[1].set_title(f'Zoomed on oscillatory behavior', size=16, pad=2)
    ax[1].annotate(text='(b)', xy=[0.35, 0.8 * 2.5], size=16)
    ax[1].set_xlim(0, 6)
    ax[1].set_ylim(0, 2.5)
    # ax.annotate(text=val, xy=[0, 0.9 * np.amax(data)], size=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig('CoherentAboveEntropy2.pdf')

"""
Paths = ['BelowRungeEntropy.npy', 'LasingRungeEntropy.npy', 'AboveRungeEntropy.npy']
cons = ['(a)', '(b)', '(c)']

for Path, val in zip(Paths, cons):
    data = parser(Path)
    plot(Path, val)
"""

os.chdir('..')
os.chdir('Coherent2')
Path = 'AboveRunge2Entropy.npy'
plot2(Path)

