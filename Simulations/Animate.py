import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
FPS = 10


def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save. Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		Data = np.load(file)
	return Data
path = '/Users/andreasevensen/Documents/GitHub/Bachelor-Thesis/Data/Euler100_100.npy'
Data = parser(path)#
cax = plt.imshow(Data[0].real, extent = [0,3 * 100,0,3 * 100], origin = 'lower', animated = False, interpolation = 'bilinear', vmax = 1e-4, vmin = 0)
def animate(n):
	Dataset = Data[n].real
	cax.set_array(Dataset)
	plt.title(f'Iteration {n}')
	return fig,

ani = FuncAnimation(fig, animate, interval = 1, frames = len(Data), blit= False)
plt.colorbar()
try:
	ani.save('Euler100_100REAL.gif', fps=FPS, writer = 'pillow', extra_args=['-vcodec', 'libx264'])
except:
	try:
		ani.save('Euler100_100REAL.gif', fps=FPS, writer = 'pillow')
	except Exception as e:
		print(e)

