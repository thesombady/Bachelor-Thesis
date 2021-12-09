import numpy as np
import matplotlib.pyplot as plt
import os

N = 100
os.chdir('test')
delta = 0.01

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}


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
                    modes[j] += data[j, m][l, n].real
    return modes


def name(path):
    if 'Above' in path:
        return 'Occupation distribution above masing threshold'
    elif 'Lasing' in path:
        return 'Occupation distribution at masing threshold'
    elif 'Below' in path:
        return 'Occupation distribution below masing threshold'


def name2(path):
    if 'Above' in path:
        return 'ModeOccupationAbove.pdf'
    elif 'Lasing' in path:
        return 'ModeOccupationLasing.pdf'
    elif 'Below' in path:
        return 'ModeOccupationBelow.pdf'
    else:
        raise TypeError("Don't work")


def plotish(path):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    modes = np.array([i for i in range(N)], dtype=int)
    data = parser(path)[-1]
    oc = occupation(data)
    ax.bar(modes, oc, width=1, label='Data')
    ax.set_xlabel(r'Photon mode $n$', size=15, labelpad=0)
    ax.set_ylabel(r'Probability', size=15, labelpad=2)
    ax.set_title(name(path), size=17)
    # plt.annotate(text='(a)', xy=[3, 0.95 * float(np.amax(oc))], size=16)
    oc1 = np.sqrt(ocu(data))
    poisson = np.array([oc1 ** (2 * n) / np.math.factorial(n) for n in range(100)], dtype=np.float128) * np.exp(-oc1 ** 2)
    plt.plot(modes, poisson, '.', markersize=5, label='Poisson', color='red')
    xlist = np.linspace(0, 100, 10000)
    gaussian = 1/np.sqrt(2 * np.pi * oc1 ** 2) * np.exp(- (xlist - oc1 ** 2) ** 2 /(2 * oc1 ** 2))
    plt.plot(xlist, gaussian, '-', markersize=1, label='Gaussian approximation', color='orange')
    # print(poisson[-1])
    plt.legend()
    plt.grid()
    # plt.savefig(name2(path))
    # plt.show()
    plt.savefig('AboveLong.pdf')


def ocu(data):
    oc = np.full((N, 3, N, 3), 0, dtype=complex)
    for index, val in np.ndenumerate(data):
        j = index[0]
        m = index[1]
        l = index[2]
        n = index[3]
        oc[j, m][l, n] = val * j
    """
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val = j * data[j, m][l, n]
                    oc[j, m][l, n] = val
    """
    return oc.reshape(3 * N, - 1, order='F').trace().real



Path = 'RungeAbove40000_100_0.01_CTrue.npy'

plotish(Path)
# print(ocu(parser(Path)[-1]))



