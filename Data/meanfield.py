import numpy as np
import os
os.chdir('Coherent')
N = 100


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)



def mean(data):
    zeros = np.full([N, 3, N, 3], 0 , dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    try:
                        zeros[j, m][l, n] += np.sqrt(l) * data[j, m][l - 1, n]
                    except:
                        zeros[j, m][l, n] += 0
    return zeros.reshape(3 * N, -1, order='F').trace()


path = 'RungeAbove10000_100_0.01_CFalse_iter10.npy'
print(mean(parser(path)[-1]))
