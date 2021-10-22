import numpy as np
import matplotlib.pyplot as plt
import os
PATH = 'EulerBelow1000_50_1Energy.npy'
N = 100
os.chdir('..')


def occupation(data):
    list1 = np.zeros(N)
    list2 = np.zeros(N)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val1 = data[j, m][l, n].real * l
                    list1[l] += val1
                    val2 = data[j, m][l, n].real * j
                    list2[j] += val2
    return (list1 + list2)/sum(list1 + list2)


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def plot(data):
    fig, ax = plt.subplots()
    xlist = [i for i in range(len(data))]
    ax.bar(xlist, data, label='Average photon occupancy')
    ax.set_title('Average photon occupancy, above lasing threshold.')
    plt.legend()
    plt.savefig('test.png')
    plt.show()
    plt.clf()

DATA = parser(PATH)
plot(DATA)

