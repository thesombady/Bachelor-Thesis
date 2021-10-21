import numpy as np
import matplotlib.pyplot as plt
import os

PATH = 'EulerAbove1000_100_2Energy.npy'
KEY = '_2Energy'
deltas = 0.0001
os.chdir('..')


def parser(path):
	"""A parser function, which utilizes the functionality of numpy.save.
	Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		data = np.load(file)
	return data


def name(path):
	if "Lasing" in path:
		return '37.5'
	elif "Above" in path:
		return '150'
	elif "Below" in path:
		return '34'
	else:
		raise KeyError(f"No key found")


def name2(path):
	path2 = path.replace('.npy', '.png')
	path3 = os.path.join(os.path.join(os.getcwd(), 'Sim'), path2)
	return path3


def plot3(data, path, methodname):
	xlist = [i * deltas for i in range(len(data))]
	plt.plot(xlist, data, '.', markersize=0.8, label='Average Energy')
	plt.legend()
	plt.title(r'{Method:}$<E> = Tr[\rho H]$, $\omega_f ={omega:}$'.format(Method=methodname,omega=name(path)))
	plt.xlabel('Time, a.u')
	plt.ylabel('Energy, a.u')
	plt.savefig(name2(path))
	plt.clf()


# plot3(parser(PATH), PATH)
"""
Values = []
with os.scandir() as it:
	for entry in it:
		if KEY in entry.name:
			if 'Euler' in entry.name:
				plot3(parser(entry.name), entry.name, 'Euler-Method')
			elif 'Runge' in entry.name:
				plot3(parser(entry.name), entry.name, 'Runge-Kutta method')
			else:
				raise TypeError('No method found')
"""

PATH = 'EulerAbove1000_100_1Energy.npy'
plot3(parser(PATH), PATH, 'Eulers method')
