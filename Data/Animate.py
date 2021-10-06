import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
FPS = 10


def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save. Using this method,
	each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		data = np.load(file)
	return data


Path = os.path.join(os.getcwd(), 'EulerAbove100_3.npy')
Data = parser(Path)
Shape = 3 * 3
cax = plt.imshow(Data[0].reshape(Shape, - 1).real, extent=[0, Shape, 0, Shape], origin='lower',
				animated=False, interpolation='bilinear')  # , vmax = 1e-4, vmin = 0)


def animate(n):
	dataset = Data[n].reshape(Shape, - 1).real
	cax.set_array(dataset)
	plt.title(f'Iteration {n}')
	return fig,


ani = FuncAnimation(fig, animate, interval=1, frames=len(Data))
plt.colorbar()
name = 'EulerAbove100_3REAL.gif'
path = os.path.join(os.path.join(os.getcwd(), 'Sim', name))
try:
	ani.save(path, fps=FPS, writer='pillow', extra_args=['-vcodec', 'libx264'])
except Exception as e:
	try:
		ani.save(path, fps=FPS, writer='pillow')
	except Exception as e:
		print(e)
