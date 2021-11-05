import numpy as np
import matplotlib.pyplot as plt
import os
PATH = 'EulerAbove250_25_0.04.npy'
PATH = os.path.join(os.getcwd(), PATH)
N =25


def occupation(data):
    zero = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    if n == 1 or m == 1:
                        zero[j, m][l, n] = data[j, m][l, n] * l
                    else:
                        zero[j, m][l, n] = 0
    return zero.reshape(3 * N, -1, order='F').trace().real


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def plot2(data):
    xlist = np.array([i for i in range(len(data))])
    plt.plot(xlist, data, '.')
    plt.xlabel(r'n')
    plt.show()


def plot(data, xlist):
    fig, ax = plt.subplots()
    xlist = [i for i in range(len(data))]
    ax.bar(xlist, data, label='Average photon occupancy')
    ax.set_title('Average photon occupancy, below lasing threshold.')
    ax.set_ylim(0,1.05)
    plt.legend()
    # plt.savefig('test.png')
    plt.show()
    plt.clf()


def plotmaker(data):
    dataset = [occupation(data[i]) for i in range(len(data))]
    plot(dataset, [i for i in range(N)])

DATA = parser(PATH)
plotmaker(DATA)
