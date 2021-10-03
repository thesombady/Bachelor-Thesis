import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
FPS = 1


def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save. Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		Data = np.load(file)
	return Data

def animate(n):
	Dataset = Data[n]
	cax.set_array(Dataset.real)
global Data
#path = os.path.join(os.getcwd(), 'test.npy')
path = '/Users/andreasevensen/Documents/GitHub/Bachelor-Thesis/Data/Runge100_3.npy'
Data = parser(path)#
cax = plt.imshow(Data[-1].real, extent = [0,3,0,3], origin = 'lower', animated = False, interpolation = 'bilinear')
ani = FuncAnimation(fig, animate, interval = 1, frames = len(Data), blit= True)
plt.colorbar()
"""
try:
	ani.save('Test.gif', fps=FPS, writer = 'pillow', extra_args=['-vcodec', 'libx264'])
except:
	ani.save('Test.gif', fps=FPS, writer = 'pillow')
"""
try:
	ani.save('test.mp4')
except Exception as e:
	raise(e)