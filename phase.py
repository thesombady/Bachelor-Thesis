import numpy as np
import matplotlib.pyplot as plt
N = 20
Size = 3 * N

w_2 = 0

def parser(path) -> np.array:
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    if 'Above' in path:
        w_2 = 150
    elif 'Lasing' in path:
        w_2 = 37.5
    elif 'Below' in path:
        w_2 = 34
    else:
        raise TypeError('Could not match the top-level')
    return data1


gamma_h = 1
hbar = 1
g = 5 * gamma_h
w_f = 30 * gamma_h
w_0 = 0
w_1 = w_f
w = [w_0, w_1, w_2]


def delta(n1, n2):
    if n1 == n2:
        return 1
    else:
        return 0


def energy(data1, n) -> float:
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


def energysquared(data, n) -> float:
    """Returns <E^2> = Tr[rho H^2]"""

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

    rho = data[n]
    rho0 = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val = hbar ** 2 * (
                        w_0 ** 2 * rho[j, m][l, 0] + w_1 ** 2 * rho[j, m][l, 1] + w_2 ** 2 * rho[j, m][l, 2]
                        + w_f * l * (w_0 * rho[j, m][l, 0] + w_1 * rho[j, m][l, 1] + w_2 * rho[j, m][l, 2])
                        + g * (np.sqrt(l + 1) * delta(n, 1) * rho[j, m][l + 1, 0] * (w_0 + w_1 + w_2)
                               + np.sqrt(l) * delta(n, 0) * minimum(j, m, l - 1, 1) * (w_0 + w_1 + w_2))
                        + w_f ** 2 * l ** 2 * rho[j, m][l, n]
                        + w_f * g * (np.sqrt(l + 1) * (l + 1) * delta(n, 1) * rho[j, m][l + 1, 0]
                                   + np.sqrt(l) * (l - 1) * delta(n, 0) * rho[j, m][l - 1, 1])
                        + g * (w_0 + w_1 + w_2) * (delta(n, 1) * np.sqrt(l + 1) * (l + 1) * maximum(j, m, l + 1, 0)
                               + delta(n, 0) * np.sqrt(l) * (l - 1) * minimum(j, m, l - 1, 1))
                        + w_f * g * l * (np.sqrt(l + 1) * delta(n, 1) * maximum(j, m, l + 1, 0)
                                         + np.sqrt(l) * delta(n, 0) * minimum(j, m, l - 1, 1))
                        + g ** 2 * (l * delta(n, 0) * rho[j, m][l, 0]
                                    + (l + 1) * delta(n, 1) * rho[j, m][l, 1])
                    )
                    rho0[j, m][l, n] = val

    return rho0.reshape(Size, -1, order='F').trace()


def alpha(rho):
    k = 1
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

    rho0 = np.full((N, 3, N, 3), 0, dtype = float)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val1 = k * (
                        maximum(j + 1, m, l, n) * np.sqrt(j + 1) + minimum(j - 1, m, l, n)
                    )
                    val2 = k * (
                        maximum(j + 1, m, l, n) * np.sqrt(j + 1) - minimum(j - 1, m, l, n)
                    )
                    rho0[j, m][l, n] = val1.real + val2.imag
    return plot(rho0)


def plot(data):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(data.reshape(Size, - 1, order='F'), extent=(0, 3 * N, 0, 3 * N), origin='lower', animated=False,
              cmap='cividis')
    plt.show()

Path = 'EulerAbove1000_20_0.1.npy'
alpha(parser(Path)[-600])
