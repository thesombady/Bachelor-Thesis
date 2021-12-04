import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('..')


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def find(path):
    if '0.01' in path:
        return 0.01
    elif '0.02' in path:
        return 0.02
    elif '0.001' in path:
        return 0.001
    elif '0.005' in path:
        return 0.005
    else:
        raise KeyError("No delta correspondance")


def name(path):
    if 'Runge' in path:
        return 'Runge '
    elif 'Euler' in path:
        return 'Euler '
    else:
        raise KeyError("No method correspondance")


def plotish(path):
    data = parser(path) /10
    delta = find(path)
    xlist = np.array([i * delta for i in range(len(data))])
    ax[0].plot(xlist, data.real, '-', markersize=1, label=name(path)+str(delta))
    ax[1].plot(xlist, data.real, '-',markersize=1, label=name(path)+str(delta))



Path1 = 'Euler0.01Energy.npy'
Path2 = 'Runge0.01Energy.npy'
Path3 = 'Euler0.02Energy.npy'
Path4 = 'Runge0.02Energy.npy'
Path5 = 'Euler0.001Energy.npy'
Path6 = 'Runge0.001Energy.npy'
Path7 = 'Euler0.005Energy.npy'
Path8 = 'Runge0.005Energy.npy'
Path = [Path1, Path2, Path3, Path4, Path5,
        Path6, Path7, Path8]
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
for path in Path:
    plotish(path)
plt.grid()
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}
ax[0].set_ylim(0, 8)
ax[0].set_title(r'Average energy, $\langle E\rangle = Tr~[\hat{\rho}\hat{H}]$', size=17)
ax[1].set_xlabel(r'Time, $\gamma_h^{-1}$', size=15)
ax[0].annotate(text='(a)', xy=[0.5, 0.9 * 8],size=12)
ax[1].annotate(text='(b)', xy=[0.5, 0.95 * 5], size=12)
ax[1].set_ylim(3, 5)
ax[1].set_title('Average energy, zoomed', size=12, pad=1)
for axes in ax:
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes.set_ylabel(r'Energy, $\hbar$ eV')
plt.subplots_adjust(right=0.7)
# plt.show()
plt.savefig('EnergyComparison.pdf')