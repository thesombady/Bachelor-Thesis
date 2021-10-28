import numpy as np
import matplotlib.pyplot as plt
import os
PATH = 'Sim/RungeAbove1000_50_0.01.npy'
PATH = os.path.join(os.getcwd(), PATH)
N = 50
os.chdir('..')
os.chdir('..')


def occupation(data):
    list1 = np.zeros(N, dtype=complex)
    list2 = np.zeros(N)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    val1 = data[j, m][l, n].real * l + data[j, m][l, n].imag
                    list1[l] += val1
                    val2 = data[j, m][l, n].real * j + data[j, m][l, n].imag
                    list2[j] += val2
    return (list1 + list2)/sum(list1 + list2)


def occupation2(rho):
    val_list = np.zeros(N)
    for m in range(3):
        for j in range(N):
            for n in range(3):
                for l in range(N):
                    val = rho[j, m][l, n] * l
                    val_list[l] += val.real + val.imag
                    if val != 0j:
                        print(j, m, l, n)
                        print(val)
    print(val_list)
    return val_list



def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def plot2(data):
    xlist = np.array([i for i in range(len(data))])
    plt.plot(xlist, data, '.')
    plt.xlabel(r'n')
    plt.show()


def plot(data):
    fig, ax = plt.subplots()
    xlist = [i for i in range(len(data))]
    ax.bar(xlist, data, label='Average photon occupancy')
    ax.set_title('Average photon occupancy, below lasing threshold.')
    ax.set_ylim(0,1.05)
    plt.legend()
    # plt.savefig('test.png')
    plt.show()
    plt.clf()

DATA = parser(PATH)
occupation2(DATA[50])
