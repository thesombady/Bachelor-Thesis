import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PATH = os.getcwd()
Name = 'EulerAbove1000_100_1.npy'
N = 100
Shape = 3 * N


def parser(path) -> np.array:
    with open(path, 'rb') as file:
        data = np.load(file)
    return data


def plot(data):
    dataset = data.reshape(Shape, -1).real
    plt.imshow(dataset, extent=[0, Shape, 0, Shape], interpolation='bilinear',
			origin='lower', animated=False, vmin=0, vmax=1)
    plt.show()


fig, ax = plt.subplots(2,1, figsize=(20, 20))  # plt.figure()
FPS = 50

path2 = os.path.join(PATH, Name)
data1 = parser(path2)
cax1 = ax[0].imshow(data1[0].reshape(Shape, - 1).real, extent=[0, Shape, 0, Shape],
                 interpolation='bilinear', origin='lower', animated=False, vmin=0, vmax=1)
cax2 = ax[1].imshow(data1[0].reshape(Shape, - 1).imag, extent=[0, Shape, 0, Shape],
                 interpolation='bilinear', origin='lower', animated=False, vmin=0, vmax=data1[-1].imag.sum())

c1 = plt.colorbar(cax1, ax=ax[0], orientation='horizontal', fraction=0.04)
c2 = plt.colorbar(cax2, ax=ax[1], orientation='horizontal', fraction=0.04)


def animate(n):
    dataset = data1[n].reshape(Shape, - 1)
    fig.suptitle(f'Iteration {n}')
    im1 = cax1.set_array(dataset.real)
    ax[0].set_title('Real')
    im2 = cax2.set_array(dataset.imag)
    ax[1].set_title('Imaginary')
    print(n, dataset.imag.sum())
    return fig,


ani = FuncAnimation(fig, animate, interval=1, frames=len(data1))


def name(path2):
    name1 = path2.replace('.npy', '.gif')
    return name1


try:
    ani.save(name(path2), fps=FPS, writer = 'pillow', extra_args=['-vcodec', 'libx264'])
except:
    ani.save(name(path2), fps=FPS, writer = 'pillow')


