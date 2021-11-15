import numpy as np
import matplotlib.pyplot as plt
N = 5


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def ocu(data):
    oc = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val = j * data[j, m][l, n]
                    oc[j, m][l, n] = val
    return oc.reshape(3 * N, -1, order='F').trace().real




Path = 'RungeAbove10000_5_0.01.npy'
data = parser(Path)[-1]
print(ocu(data))