import numpy as np
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()
os.chdir('uniruns')
Name = 'EulerAbove10000_4_0.001_CFalse_iter.npy'
deltas = 0.01
# path1 = os.path.join(PATH, Name)
N = 4
Shape = 3 * N
gamma_h = 1
gamma_c = 1
hbar = 6.58 * 10 ** (-6)
g = 1
w_0 = 0
w_1 = 30 * gamma_h
w_f = w_1
# w_2 is defined beneath


def omega(name):
    """Returns the correct w_2 frequency, given the file-name."""
    if 'Lasing' in name:
        print('Lasing')
        return 37.5
    if 'Below' in name:
        print('Below')
        return 34
    if 'Above' in name:
        print('Above')
        return 150
    else:
        raise KeyError("Can't determine what omega to use")


w_2 = omega(Name) * gamma_h
w = [w_0, w_f, w_2]
# FWHM -> gamma_alpha * (n_alpha + 1); alpha in {c,h} # Alex Kahlee


def parser(path) -> np.array:
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    return data1


def delta(n_1, n_2) -> int:
    """Kronicka-delta function, yields 1 if indexing is the same, else zero."""
    if n_1 == n_2:
        return 1
    else:
        return 0


def energy2(data1, n) -> float:
    """Returns the energy, in terms of <E> = Tr(Rho * H)"""
    rho = data1[n]
    rho0 = np.full((N, 3, N, 3), 0, dtype=complex)

    def maximum(p, s, k, d) -> complex:
        """Returning the null-state
        a^\dagger|a> = 0 * |null> if a is the boundary."""
        if p == N or k == N:
            return 0
        else:
            return rho[p, s][k, d]

    def minimum(p, s, k, d) -> complex:
        """Returning the null-state a|0> = 0 * |null>."""
        if p == -1 or k == -1:
            return 0
        else:
            return rho[p, s][k, d]

    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val = hbar * (
                        w_0 * rho[j, m][l, 0] + w_1 * rho[j, m][l, 1] + w_2 * rho[j, m][l, 2]
                        + w_f * l * rho[j, m][l, n]
                        + g * (
                            np.sqrt(l + 1) * delta(n, 1) * maximum(j, m, l + 1, 0)
                            + np.sqrt(l) * delta(n, 0) * minimum(j, m, l - 1, 1)
                        )
                    )
                    rho0[j, m][l, n] = val
    return rho0.reshape(3 * N, - 1, order='F').trace()


def energy(data1, n) -> complex:
    """Returns the energy, in terms of E = Tr(Rho * H)"""
    dataset = data1[n]
    rho0 = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val = dataset[j, m][l, n]
                    for k in range(3):  # This is the Hamiltonian part
                        val1 = val * w[k] * delta(n, k) * hbar
                        rho0[j, m][l, k] = val1
                    val2 = l * hbar * val *  w_f
                    rho0[j, m][l, n] = val2
                    if l == 0:
                        val3 = 0  # could pass
                        val4 = np.sqrt(l + 1) * delta(n, 1) * g * hbar * val
                        rho0[j, m][l + 1, 0] = val4
                    elif l == N - 1:
                        val4 = 0  # could pass
                        val3 = np.sqrt(l) * delta(n, 0) * g * hbar * val
                        rho0[j, m][l - 1, 1] = val3
                    else:
                        val3 = np.sqrt(l) * delta(n, 0) * g * hbar * val
                        rho0[j, m][l - 1, 1] = val3
                        val4 = np.sqrt(l + 1) * delta(n, 1) * g * hbar * val
                        rho0[j, m][l + 1, 0] = val4
    return rho0.reshape(3 * N, - 1).trace()


def entropy(data1, n) -> complex:
    """Computes the Von-Nuemann entropy of the density matrix"""
    dataset = data1[n]
    data2 = dataset * np.log(dataset)
    return data2.reshape(3 * N, - 1).trace()


def name1(path2):
    name1 = path2.replace('.npy', '.png')
    return name1


def name3(path2):
    name1 = path2.replace('.npy', 'Energy.npy')
    return name1


def name2(path):
    list1 = ['Euler', 'Runge']
    list2 = ['Below', 'Above', 'Lasing']
    for val1 in list1:
        if val1 in path:
            for val2 in list2:
                if val2 in path:
                    return 'Energy Energy using {} method, at {}-threshold'.format(val1, val2.lower())
    return 'nothing'


data = []
iterations = 0
for i in range(1, 11):
    path = f'RungeAbove10000_4_0.001_CFalse_iter{i}.npy'
    rho = parser(path)
    data.append(rho)
    iterations += len(rho)

# data = np.array(data, dtype=object)
rawdata = np.array([data[i][j] for i in range(len(data)) for j in range(len(data[i]))])
for i in range(len(rawdata)):
    for j in range(len(rawdata)):
        if rawdata[i] == rawdata[j] and i != j:
            print('Duplicate')


"""
energyval = []
for i in range(len(rawdata)):
    energyval.append(energy(rawdata, i))
with open('testEnergy2.npy', 'wb') as file:
    np.save(file, np.array(energyval))
"""
"""
for i in range(len(data)):
    vals = energy2(data, i)
    print(vals, i)
    value.append(vals.real)

with open(name3(path1), 'wb') as file:
    np.save(file, np.array(value))
"""
