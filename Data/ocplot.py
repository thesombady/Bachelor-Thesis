import numpy as np
import matplotlib.pyplot as plt
import os

N = 100
os.chdir('uniruns')


def parser(path):
    """Standard parser that imports a .npy file.
    Does not parse the content but rather the file."""
    with open(path, 'rb') as file:
        return np.load(file)


def occupation(data):
    modes = np.array([0 for i in range(N)], dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    if m == 1 and n == 1:
                        modes[j] += data[j, m][l, n].real
                    # modes[j] += data[j, m][l, n]
    plotish(modes)


def plotish(data1):
    alpha = 11.68
    modes = np.array([i for i in range(N)], dtype=int)
    plt.bar(modes, data1, width=1, label='Simulated values')
    plt.xlabel(r'Photon mode $n$')
    plt.ylabel(r'Probability')
    plt.title('Occupation of the photon modes')
    plt.legend()
    plt.grid()
    plt.show()


# os.chdir('/Users/andreasevensen/Documents/GitHub/Bachelor-Thesis/Data/uniruns/')
""" # Probability distribution above the masing threshold.
rawdata = []
for i in range(1, 11):
    path = f'RungeAbove10000_100_0.01_CFalse_iter{i}.npy'
    rawdata.append(parser(path))

data = np.array([rawdata[i][-1] for i in range(len(rawdata))])[-1]
occupation(data)
"""
path = 'RungeLasing10000_100_0.01_CFalse_iter10.npy'
data = parser(path)[-1]
occupation(data)

