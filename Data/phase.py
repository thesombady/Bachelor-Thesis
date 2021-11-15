import numpy as np
import matplotlib.pyplot as plt
N = 5
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


gamma_h = 1  # 10 ** (-8)  # might have to use something similar
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

    rhox = np.full((N, 3, N, 3), 0, dtype=complex)
    rhop = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    valx = (
                        minimum(j - 1, m, l, n) * np.sqrt(j)
                        #  + maximum(j + 1, m , l, n) * np.sqrt(j + 1)
                    )
                    valp = (
                        maximum(j + 1, m, l, n) * np.sqrt(j + 1)
                        #  - minimum(j - 1, m, l, n) * np.sqrt(j)
                    )

                    rhox[j, m][l, n] = valx
                    rhop[j, m][l, n] = valp
    return plotish2(rhox.reshape(3 * N, - 1, order='F'), rhop.reshape(3 * N, - 1, order='F'))


def plotish(rhox, rhop):
    rhox = rhox.real
    rhop = rhop.imag
    fig, ax = plt.subplots(1, 3, figsize=(6, 6))
    cax1 = ax[0].imshow(rhox, extent=[0, 3 * N, 0, 3 * N], origin='lower',
                     interpolation='none', cmap='Greys')
    cax2 = ax[1].imshow(rhop, extent=[0, 3 * N, 0, 3 * N], origin='lower',
                        interpolation='none', cmap='Greys')
    cax3 = ax[2].imshow((rhop + rhox), extent=[0, 3 * N, 0, 3 * N], origin='lower',
                        interpolation='none', cmap='Greys')
    plt.colorbar(cax1, ax=ax[0], orientation='horizontal')
    plt.colorbar(cax2, ax=ax[1], orientation='horizontal')
    plt.colorbar(cax3, ax=ax[2], orientation='horizontal')
    print(rhox.trace(), rhop.trace())
    plt.show()


def alpha2(rho):
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

    rhox = np.full((N, 3, N, 3), 0, dtype=complex)
    rhop = np.full((N, 3, N, 3), 0, dtype=complex)
    rhoa = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    valx = (
                        minimum(j - 1, m, l, n) * np.sqrt(j)
                        + maximum(j + 1, m , l, n) * np.sqrt(j + 1)
                    )
                    valp = (
                        maximum(j + 1, m, l, n) * np.sqrt(j + 1)
                        - minimum(j - 1, m, l, n) * np.sqrt(j)
                    )
                    val = (
                        j * rho[j, m][l, n]
                    )

                    rhox[j, m][l, n] = valx
                    rhop[j, m][l, n] = valp
                    rhoa[j, m][l, n] = val

    return plotish2(rhoa.reshape(3 * N, - 1, order='F'))


def plotish2(rho):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    cax1 = ax.imshow(rho.real + rho.imag, extent=[0, 3 * N, 0, 3 * N], origin='lower',
                        interpolation='none', cmap='Greys')
    plt.colorbar(cax1, ax=ax, orientation='horizontal')
    plt.show()



Path = 'RungeAbove10000_5_0.01.npy'
# for i in range(len(parser(Path))):
alpha2(parser(Path)[4000])

