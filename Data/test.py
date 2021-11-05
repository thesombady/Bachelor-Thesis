import numpy as np
import matplotlib.pyplot as plt
import os

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
                else:
                    raise KeyError(
                        'Could not match delta'
                    )
            else:
                pass
    assert len(files) == len(deltas), 'missmatch'
    for i in range(len(files)):
        plotmax(parser(files[i]), deltas[i], files[i])


findfiles()
