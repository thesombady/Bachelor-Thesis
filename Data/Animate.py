import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# os.chdir('..')
PATH = os.getcwd()
Name = 'RungeAbove1000_5_0.01.npy'
N = 5
Shape = 3 * N
delta = 0.01


def parser(path) -> np.array:
    with open(path, 'rb') as file:
        data = np.load(file)
    return data


def plot(data):
    dataset = data.reshape(Shape, -1).real
    plt.imshow(dataset, extent=[0, Shape, 0, Shape],
            origin='lower', animated=False, vmin=0, vmax=1)
    plt.show()


fig, ax = plt.subplots(2, 1, figsize=(20, 20))
FPS = 50
os.chdir('..')

path2 = os.path.join(PATH, Name)
datalist = []
for i in range(1, 11):
    path = f'RungeAbove1000_5_0.01_CFalse_iter{i}.npy'
    datalist.append(parser(path))

data1 = np.array([datalist[i][j] for i in range(len(datalist)) for j in range(len(datalist[i]))])


#data1 = parser(path2)

val = data1[-1].reshape(Shape, - 1, order='F')
cax1 = ax[0].imshow(val.real, extent=[0, Shape, 0, Shape], interpolation='none',
                origin='lower', animated=False, vmin=0, vmax=1, cmap='Greys')
cax2 = ax[1].imshow(val.imag, extent=[0, Shape, 0, Shape],
                origin='lower', animated=False, interpolation='none',
                vmin=np.amin(val.imag), vmax=np.amax(val.imag), cmap='Greys')

c1 = plt.colorbar(cax1, ax=ax[0], orientation='horizontal', fraction=0.04)
c2 = plt.colorbar(cax2, ax=ax[1], orientation='horizontal', fraction=0.04)


def animate(n):
    n = n
    dataset = data1[n].reshape(Shape, - 1, order='F')
    fig.suptitle(f'Time {n * delta}')
    im1 = cax1.set_array(dataset.real)
    ax[0].set_title('Real')
    im2 = cax2.set_array(dataset.imag)
    ax[1].set_title('Imaginary')
    print(n)
    return fig,


def name(path):
    name1 = path.replace('.npy', '.gif')
    return name1


ani = FuncAnimation(fig, animate, interval=1, frames=int(len(data1)))
try:
    ani.save(name(path2), fps=FPS, writer='pillow', extra_args=['-vcodec', 'libx264'])
except Exception as E:
    ani.save(name(path2), fps=FPS, writer='pillow')
