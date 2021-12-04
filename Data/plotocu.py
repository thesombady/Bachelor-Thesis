import numpy as np
import os
import matplotlib.pyplot as plt

delta = 0.01

os.chdir('Coherent2')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}

# plt.rc('font', **font)


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def name(path):
    if 'Above' in path:
        return 'above masing threshold'
    elif 'Lasing' in path:
        return 'at masing threshold'
    elif 'Below' in path:
        return 'below masing threshold'
    else:
        raise TypeError("Can't match the path")


def name2(path):
    if 'Above' in path:
        return 'OccupationAbove.pdf'
    elif 'Lasing' in path:
        return 'OccupationLasing.pdf'
    elif 'Below' in path:
        return 'OccupationBelow.pdf'
    else:
        raise TypeError("Can't match the path")


def plot(path):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    data = parser(path)
    xarray = np.array([i * delta for i in range(len(data))])
    ax.plot(xarray, data, '-', markersize=1)
    ax.grid()
    ax.set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=0)
    ax.set_ylabel(r'Average occupation, $\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax.set_title(f'Average occupation when the system\nis operating {name(path)}', size=17)
    ax.annotate(text='(a)', xy=[-0.2, 0.95 * float(data[-1])], size=16)
    # plt.show()
    # plt.savefig(name2(path))
    plt.show()



def plot2(path):
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5))
    data = parser(path)
    xarray = np.array([i * delta for i in range(len(data))])
    ax[0].plot(xarray, data, '-', markersize=1)
    ax[0].grid()
    ax[0].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=4)
    ax[0].set_ylabel(r'$\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax[0].set_title(f'Average occupation when the system\nis operating {name(path)}', size=17)
    ax[0].annotate(text='(a)', xy=[-0.2, 0.9 * float(data[-1])], size=16)
    ax[1].plot(xarray, data,'-', markersize=1)
    ax[1].grid()
    ax[1].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=0)
    ax[1].set_ylabel(r'$\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax[1].set_title(f'Zoomed on Rabi-oscillation', size=16)
    ax[1].annotate(text='(b)', xy=[0.1, 0.9 * 6], size=16)
    ax[1].set_xlim(0, 2)
    ax[1].set_ylim(3, 6)
    plt.tight_layout()
    # plt.show()
    plt.savefig('CoherentAbove.pdf')


Path = 'CoherentAbove.npy'
plot2(Path)
