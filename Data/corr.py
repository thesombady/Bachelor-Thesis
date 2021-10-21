import numpy as np
import os
import matplotlib.pyplot as plt

PATH = os.getcwd()
Name = 'EulerAbove1000_100_1.npy'
deltas = 0.0001
path1 = os.path.join(PATH, Name)
N = 100
Shape = 3 * N
gamma_h = 1
gamma_c = 1
w_0 = 0
w_1 = 30 * gamma_h
w_f = w_1
# w_2 = 37.5 * gamma_h  # 34, 37.5, 150; This is the one we change depending on the cavity state


def omega(name):
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
hbar = 1
g = 1

# FWHM -> gamma_alpha * (n_alpha + 1); alpha in {c,h} # Alex Kahlee


def parser(path) -> np.array:
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    return data1


def delta(n_1, n_2) -> int:
    if n_1 == n_2:
        return 1
    else:
        return 0


def energy(data1, n) -> complex:
    """Returns the energy, in terms of E = Tr(Rho * H)"""
    dataset = data1[n]
    rho0 = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    for i in range(3):  # This is the Hamiltonian part
                        val = dataset[j, m][l, n]
                        val1 = val * w[i] * delta(n, i) * hbar
                        val2 = l * hbar * val
                        rho0[j, m][l, n] = val1 + val2
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
    return rho0.reshape(3 * N, -1).trace()


def entropy(data1, n) -> complex:
    """Computes the Von-Nuemann entropy of the density matrix"""
    dataset = data1[n]
    data2 = dataset * np.log(dataset)
    return data2.reshape(3 * N, -1).trace()


def name(path2):
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


data = parser(path1)
value = []
xlist = [i * deltas for i in range((len(data)))]
for i in range(len(data)):
    vals = energy(data, i)
    print(vals, i)
    value.append(vals.real)

with open(name3(path1), 'wb') as file:
    np.save(file, np.array(value))


plt.plot(xlist, value, '.', label='Average energy', markersize=0.8)
plt.title(f'{name2(path1)}')
plt.xlabel('Time, a.u')
plt.ylabel('Energy, a.u')
plt.legend()
plt.savefig(name(path1))
