import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=1, suppress=True, threshold=81)


N = 3
Size = 3 * N


def parser(path) -> np.array:
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    return data1


def plotmax(data, deltas, path):
    fig, ax = plt.subplots(2, 1, figsize=(5, 5))
    xlist = np.array([i * deltas for i in range(len(data))])
    realmaxes = np.array([np.amax(data[i]).real for i in range(len(data))])
    imagmaxes = np.array([np.amax(data[i]).imag for i in range(len(data))])
    ax[0].plot(xlist, realmaxes, '-', markersize=1, label='Real', color='Blue')
    ax[1].plot(xlist, imagmaxes, '-', markersize=1, label='Imag', color='Red')
    fig.legend(loc=2)
    fig.tight_layout(pad=3.0)
    for axes in ax:
        axes.set_xlabel(r'Time, $\Delta t = {}$'.format(deltas))
    plt.title(path)
    plt.show()


def parsname(path):
    if 'Euler' in path:
        return 'Euler'
    else:
        return 'Runge'


def plotmax2(datas, deltas):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for data in datas:
        xlist = np.array([i * deltas for i in range(len(parser(data)))], dtype=object)
        data2 = parser(data).real
        maxreal = np.array([np.amax(data2[i]) for i in range(len(data2))], dtype=object)
        #  ax.plot(xlist, maxreal, '-', markersize=1, label=parsname(data)+str(deltas))
        print(xlist[-1])
    # plt.legend()
    # plt.show()



def findfiles():
    files = []
    deltas = []
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.npy'):
            if 'Energy' not in filename:
                files.append(filename)
                if '0.04' in filename:
                    deltas.append(0.04)
                elif '0.01' in filename:
                    deltas.append(0.01)
                elif '0.02' in filename:
                    deltas.append(0.02)
                elif '0.001' in filename:
                    deltas.append(0.001)
                elif '0.05' in filename:
                    deltas.append(0.05)
                elif '0.1' in filename:
                    deltas.append(0.1)
                else:
                    raise KeyError(
                        'Could not match delta'
                    )
            else:
                pass
    assert len(files) == len(deltas), 'missmatch'
    plotmax2(files, deltas)


def occupation(rho):
    zero = np.full((N, 3, N, 3), 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    zero[j, m][l, n] = j * rho[j, m][l, n]
    return zero.reshape(3 * N, - 1, order='F').trace()



# findfiles()
"""
Path = 'EulerAbove2000_3_0.01.npy'
data = parser(Path)[1225]  # .reshape(Size, - 1, order='F')
for j in range(3):
    for m in range(3):
        for l in range(3):
            for n in range(3):
                if data[j, m][l, n].real > 1:
                    print((j, m), (l, n))
"""
Path = 'EulerAbove1000_20_0.1.npy'
Data = parser(Path)
vals = [occupation(Data[i]) for i in range(len(Data))]
for val in vals:
    print(val)

