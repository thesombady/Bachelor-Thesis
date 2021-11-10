import numpy as np
import matplotlib.pyplot as plt
import os
#PATH = 'EulerAbove2000_3_0.02Energy.npy'
#PATH2 = 'EulerBelow2000_3_0.02Energy.npy'
#PATH3 = 'EulerLasing2000_3_0.02Energy.npy'
PATH = 'EulerAbove2000_3_0.02_2Energy.npy'
PATH = os.path.join(os.getcwd(), PATH)
N = 3


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


def plot2(data1):  # , data2, data3):
    xlist = np.array([i for i in range(len(data1))])
    plt.plot(xlist, data1, '-', markersize=1, label='Above')
    # plt.plot(xlist, data2, '-', markersize=1, label='Below')
    # plt.plot(xlist, data3, '-', markersize=1, label='Lasing')
    plt.xlabel(r'n')
    plt.legend()
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


def plotmaker(data, data2=None, data3=None):
    # dataset = [occupation(data[i]) for i in range(len(data))]
    # plot(dataset, [i for i in range(N)])
    try:
        if not data2.any() == None and not data3.any() == None:
            plot2(data, data2, data3)
    except:
        try:
            plot2(data)
        except Exception as E:
            raise E



DATA = parser(PATH)
# DATA2 = parser(PATH2)
# DATA3 = parser(PATH3)
#plotmaker(DATA, DATA2, DATA3)
plotmaker(DATA)
