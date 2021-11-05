import numpy as np
import matplotlib.pyplot as plt
N = 3
Size = 3 * N


def parser(path) -> np.array:
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    return data1

def energy(data1, n) -> complex:
    """Returns the energy, in terms of <E> = Tr(Rho * H)"""
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
    return rho0.reshape(3 * N, - 1, order='F').trace()