import numpy as np
import os

N = 3
gamma_h = 1  # usually in units of 10^(-12) seconds
gamma_c = 1  # usually in units of 10^(-12) seconds
hbar = 6.58 * 10 ** (-6)  # In units of eV/ns and the conversion of gamma_h  # 1.0545718 * 10 ** (-34)#m^2kg/s

K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h  # Lasing angular frequency
w_0 = 0
w_1 = w_f
w_2 = 150  # This parameter can change depending on mode of operation.
w_2 = w_2 * gamma_h  # This is the one we change for laser, 34, 37.5, 150 respectively.
omega = np.array([w_0, w_1, w_2])


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def delta(n1, n2) -> int:
    if n1 == n2:
        return 1
    else:
        return 0


def name3(path2):
    name1 = path2.replace('.npy', 'Energy.npy')
    return name1


def energy(data1) -> complex:  # , n) -> complex:
    """Returns the energy, in terms of E = Tr(Rho * H)"""
    # dataset = data1[n]
    dataset = data1
    rho0 = np.full((N, 3, N, 3), 0, dtype=complex)

    def maximum(p, s, k, d) -> complex:
        """Returning the null-state
        a^\dagger|a> = 0 * |null> if a is the boundary."""
        if p == N or k == N:
            return 0
        else:
            return dataset[p, s][k, d]

    def minimum(p, s, k, d) -> complex:
        """Returning the null-state a|0> = 0 * |null>."""
        if p == -1 or k == -1:
            return 0
        else:
            return dataset[p, s][k, d]

    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):  # Note, 1 is replaced with hbar
                    val = 1 * (
                        w_0 * dataset[j, m][l, 0]
                        + w_1 * dataset[j, m][l, 1]
                        + w_2 * dataset[j, m][l, 2]
                        + l * w_f * dataset[j, m][l, n]
                        + g * (
                            delta(n, 1) * maximum(j, m, l + 1, 0) * np.sqrt(l + 1)
                            + delta(n, 0) * minimum(j, m, l - 1, 1) * np.sqrt(l)
                        )
                    )
                    rho0[j, m][l, n] += val
    return rho0.reshape(3 * N, - 1, order='F').trace()


os.chdir('..')
Runge0_01 = []
for i in range(1, 3):
    path = f'RungeAbove1000_5_0.01_CFalse_iter{i}.npy'
    Runge0_01.append(parser(path))

Runge0_01 = np.array([Runge0_01[i][j] for i in range(len(Runge0_01))
                 for j in range(len(Runge0_01[i]))])

Runge0_02 = []
for i in range(1, 3):
    path = f'RungeAbove500_5_0.02_CFalse_iter{i}.npy'
    Runge0_02.append(parser(path))

Runge0_02 = np.array([Runge0_02[i][j] for i in range(len(Runge0_02))
                 for j in range(len(Runge0_02[i]))])

Runge0_001 = []
for i in range(1, 11):
    path = f'RungeAbove10000_5_0.001_CFalse_iter{i}.npy'
    Runge0_001.append(parser(path))

Runge0_001 = np.array([Runge0_001[i][j] for i in range(len(Runge0_001))
                 for j in range(len(Runge0_001[i]))])

Runge0_005 = []
for i in range(1, 3):
    path = f'RungeAbove2000_5_0.005_CFalse_iter{i}.npy'
    Runge0_005.append(parser(path))

Runge0_005 = np.array([Runge0_005[i][j] for i in range(len(Runge0_005))
                 for j in range(len(Runge0_005[i]))])

Euler0_01 = []
for i in range(1, 3):
    path = f'EulerAbove1000_5_0.01_CFalse_iter{i}.npy'
    Euler0_01.append(parser(path))

Euler0_01 = np.array([Euler0_01[i][j] for i in range(len(Euler0_01))
                 for j in range(len(Euler0_01[i]))])

Euler0_02 = []
for i in range(1, 3):
    path = f'EulerAbove500_5_0.02_CFalse_iter{i}.npy'
    Euler0_02.append(parser(path))

Euler0_02 = np.array([Euler0_02[i][j] for i in range(len(Euler0_02))
                 for j in range(len(Euler0_02[i]))])

Euler0_001 = []
for i in range(1, 11):
    path = f'EulerAbove10000_5_0.001_CFalse_iter{i}.npy'
    Euler0_001.append(parser(path))

Euler0_001 = np.array([Euler0_001[i][j] for i in range(len(Euler0_001))
                 for j in range(len(Euler0_001[i]))])

Euler0_005 = []
for i in range(1, 3):
    path = f'EulerAbove2000_5_0.005_CFalse_iter{i}.npy'
    Euler0_005.append(parser(path))

Euler0_005 = np.array([Euler0_005[i][j] for i in range(len(Euler0_005))
                 for j in range(len(Euler0_005[i]))])

Runges = [Runge0_01, Runge0_02, Runge0_001, Runge0_005]
Eulers = [Euler0_01, Euler0_02, Euler0_001, Euler0_005]
RungeNames = ['Runge0.01Energy.npy', 'Runge0.02Energy.npy', 'Runge0.001Energy.npy', 'Runge0.005Energy.npy']
EulerNames = ['Euler0.01Energy.npy', 'Euler0.02Energy.npy', 'Euler0.001Energy.npy', 'Euler0.005Energy.npy']
for data, name in zip(Runges, RungeNames):
    array = []
    for i in range(len(data)):
        array.append(energy(data[i]))
    with open(name, 'wb') as file:
        np.save(file, np.array(array))
for data, name in zip(Eulers, EulerNames):
    array = []
    for i in range(len(data)):
        array.append(energy(data[i]))
    with open(name, 'wb') as file:
        np.save(file, np.array(array))