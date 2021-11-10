import numpy as np
import matplotlib.pyplot as plt
import os

N = 5
Shape = 3 * N


def parser(path) -> np.array:
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    return data1


dicter = {
    'Real': lambda val: val.real,
    'Imag': lambda val: val.imag
}


def plot2(data, n):
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    for ax, key in zip(axs, dicter):
        dataset = dicter[key](data[n].reshape(Shape, - 1, order='F'))
        im = ax.imshow(dataset, extent=[0, Shape, 0, Shape], interpolation='bilinear',
            origin='lower', animated=False, vmin=np.amin(dataset), vmax=np.amax(dataset))
        plt.colorbar(im, ax=ax, shrink=1, orientation='horizontal')
        ax.set_title(key)
    # path = os.path.join(f'{os.getcwd()}/{Name}', f'iter{n}.png')
    # plt.savefig(path)
    plt.show()


Path = 'EulerAbove1000_5_0.01.npy'
#plot2(parser(Path), 5001)
Data = parser(Path)[-1].real
print(np.amin(Data))