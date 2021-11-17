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
                for n in range(3):
                    val = hbar * (
                        w_0 * dataset[j, m][l, 0]
                        + w_1 * dataset[j, m][l, 1]
                        + w_2 * dataset[j, m][l, 2]
                        + l * w_f * dataset[j, m][l, n]
                        + g * (
                            delta(n, 1) * maximum(j, m, l + 1, 0) * np.sqrt(l + 1)
                            + delta(n, 0) * minimum(j, m, l - 1, 1) * np.sqrt(l)
                        )
                    )
                    rho0[j, m][l, n] = val
    return rho0.reshape(3 * N, - 1).trace()


def findfiles():
    files = []
    deltas = []
    for filename in os.listdir(os.chdir('..')):
        if filename.endswith('.npy'):
            if not 'Energy' in filename:
                files.append(filename)
                if '0.01' in filename:
                    deltas.append(0.01)
                elif '0.02' in filename:
                    deltas.append(0.02)
                elif '0.001' in filename:
                    deltas.append(0.001)
                elif '0.005' in filename:
                    deltas.append(0.005)
                else:
                    raise KeyError(
                        'Could not match delta'
                    )
        else:
            pass
    assert len(files) == len(deltas), 'missmatch'
    return files, deltas


a, b = findfiles()

for data, deltas in zip(a, b):
    rho = parser(data)
    path = name3(data)
    array = []
    for i in range(len(rho)):
        array.append(energy(rho[i]))
    with open(path, 'wb') as file:
        np.save(file, np.array(array))
