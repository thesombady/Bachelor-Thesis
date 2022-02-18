import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

delta = 0.01

os.chdir('uniruns')
# os.chdir('..')
# os.chdir('coherenttest')
# os.chdir('test')
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


def log(array, a, b ,c):
    return a * np.log(array * b) + c


def expp(array, a, b, c):
    return a - b * np.exp(- c * array)


def plot(path):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    data = parser(path)
    xarray = np.array([i * delta for i in range(len(data))])
    ax.plot(xarray, data, '-', markersize=1.5, label='Simulated data')
    # val = curve_fit(expp, xarray[5000:], data[5000:], p0=[6, 1, 0.01])[0]
    # xlist2 = np.array([i * delta for i in range(5000, len(data))])
    ax.plot(xarray, xarray * 0.1164, '--', markersize=0.5, label=r'Fit: $k\cdot t + m$')
    # ax.plot(xarray[5000:], expp(xarray, val[0], val[1], val[2])[5000:], '-', markersize=1, label=r'Fit: $a - b\cdot\exp(-c \cdot t)$')
    # ax.plot(xarray, xarray * 0.1168, '-', markersize=1, label=r'Fit: $k\cdot t + m$')
    """
    plt.plot(xarray, log(xarray, val[0], val[1], val[2]), '-', markersize=1, label=r'Fit: $a\cdot\ln(n \cdot b) + c$')
    ax.set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=0)
    ax.set_ylabel(r'Average occupation, $\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax.set_title(f'Average occupation when the system\nis operating {name(path)}', size=17)
    ax.annotate(text='(a)', xy=[-0.2, 0.95 * float(data[-1])], size=16)
    plt.ylim(0, 6)
    """
    ax.annotate(text='(a)', xy=[-0.2, 0.95 * float(data[-1])], size=16)
    # ax.annotate(text='(a)', xy=[-0.2, 0.99 * float(data[-1])], size=16)
    # ax.vlines(100, 0, 10, colors='black', linestyles='dashed', label=r'Time: $100\gamma_h^{-1}$')

    # plt.ylim(0, 6.7)
    ax.set_ylabel(r'Average occupation, $\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax.set_title(f'Average occupation when the system\nis operating {name(path)}', size=17)
    ax.set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=0)
    plt.legend(loc=4, fontsize=14)
    plt.grid()
    # print(val[0])
    #plt.show()
    plt.savefig(name2(path))


def plot2(path):
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5))
    data = parser(path)
    xarray = np.array([i * delta for i in range(len(data))])
    ax[0].plot(xarray, data, '-', markersize=1)
    ax[0].grid()
    ax[0].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=4)
    ax[0].set_ylabel(r'$\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax[0].set_title(f'Average occupation when the system\nis operating {name(path)}', size=17)
    ax[0].annotate(text='(a)', xy=[0.1, 0.9 * 11], size=16)
    ax[1].plot(xarray, data, '-', markersize=1)
    ax[1].grid()
    ax[1].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=0)
    ax[1].set_ylabel(r'$\langle\hat{a}^\dagger\hat{a}\rangle$', size=15)
    ax[1].set_title(f'Zoomed on oscillatory behaviour', size=16)
    ax[1].annotate(text='(b)', xy=[0.11, 0.9 * 10.6], size=16)
    ax[1].set_xlim(0, 2)
    ax[1].set_ylim(8, 10)
    plt.tight_layout()
    # plt.show()
    plt.savefig('CoherentAbove1.25.pdf')


#Path = 'CoherentAbove.npy'
Path = 'AboveOccupation.npy'
plot(Path)
