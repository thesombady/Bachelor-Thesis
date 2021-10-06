
import os
import numpy as np
import matplotlib.pyplot as plt


def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save.
	Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		data = np.load(file)
	return data


def plot(data):
	"""Data has to be of a matrix"""
	plt.plot(data)
	plt.show()


def plotmesh(data, n):
	plt.imshow(data, origin='upper', extent=[0, 3 * n, 3, 3 * n])  # , interpolation='bilinear')
	plt.colorbar()
	plt.show()

Path = os.path.join(os.getcwd(), 'EulerAbove100_3.npy')
Data = parser(Path)
N = 3
print(Data[10])
plotmesh(Data[10].reshape((N * 3, - 1)).real, n=N)
