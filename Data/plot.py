
import os
import numpy as np
import matplotlib.pyplot as plt

PATH = "RungeAbove1000_100_2Energy.npy"
deltas = 0.001
N = 50
Shape = 3 * N


def parser(path):
	"""A parser function, which utilizes the functionality of numpy.save.
	Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		data = np.load(file)
	return data


# fig, axs = plt.subplots(2, 1, constrained_layout=True)


def plot(data, n):
	fig.suptitle(f"Time evolution of the density matrix, iteration {n}")
	"""Data has to be of a matrix"""
	dataset = data[n].reshape(Shape, - 1)
	im1 = axs[0].imshow(dataset.real, extent=[0, Shape, 0, Shape], interpolation='bilinear',
				origin='lower', animated=False, vmin=0, vmax=1)
	axs[0].set_title(r'Real part')
	axs[0].set_xlabel('3N')
	axs[0].set_ylabel('3N')
	fig.colorbar(im1, cax=axs[0])
	im2 = axs[1].imshow(dataset.imag, extent=[0, Shape, 0, Shape], interpolation='bilinear',
				origin='lower', animated=False, vmin=-1, vmax=1)
	axs[1].set_title(r'Imaginary part')
	axs[1].set_xlabel('3N')
	axs[1].set_ylabel('3N')
	fig.colorbar(im2, cax=axs[1])
	path = os.path.join(f'{os.getcwd()}/{Name}', f'iter{n}.png')
	plt.savefig(path)


dicter = {
	'Real': lambda val: val.real,
	'Imag': lambda val: val.imag
}


def plot2(data, n):
	for ax, key in zip(axs, dicter):
		dataset = dicter[key](data[n].reshape(Shape, - 1))
		im = ax.imshow(dataset, extent=[0, Shape, 0, Shape], interpolation='bilinear',
			origin='lower', animated=False, vmin=-1, vmax=1)
		plt.colorbar(im, cax=ax)
		ax.set_title(key)
	path = os.path.join(f'{os.getcwd()}/{Name}', f'iter{n}.png')
	# plt.savefig(path)
	plt.show()


def name(path):
	if "Lasing" in path:
		return 'w_2 = 37.5'
	elif "Above" in path:
		return 'w_2 = 150'
	elif "Below" in path:
		return 'w_2 = 34'
	else:
		raise KeyError(f"No key found")


def name2(path):
	return path.replace('.npy', '.png')


def plot3(data, path):
	xlist = [i * deltas for i in range(len(data))]
	plt.plot(xlist, data, '.', scale=0.8, label='Average Energy')
	plt.legend()
	plt.title(r'$<E> = Tr[\hat{\rho}\hat{H}]$, $\omega_f ={}').format(name(path))
	plt.xlabel('Time, a.u')
	plt.ylabel('Energy, a.u')
	plt.savefig(name2(path))


def name3(path1):
	"""Used to automate the plotting difference function (plotdiff)"""
	if 'Lasing' in path1:
		return 'at lasing threshold'
	elif 'Above' in path1:
		return 'above lasing threshold'
	elif 'Below' in path1:
		return 'below lasing threshold'


def plotdiff(path1, path2):
	"""Path1 has to be of euler method, path2 has to be of runge-kutta method"""
	fig, ax = plt.subplots()
	values1 = parser(path1)
	values2 = parser(path2)
	assert len(values1) == len(values2), 'Different lengths'
	xlist = np.array([i for i in range(len(values1))]) * deltas
	ax.plot(xlist, values1, '.', markersize=0.8, color='red', label="Euler")
	ax.plot(xlist, values2, '.', markersize=0.8, color='blue', label="Runge-Kutta")
	ax.set_xlabel(r'Time, a.u')
	ax.set_ylabel(r'Energy, a.u')
	ax.set_title('Average energy comparison of\nEuler-, and Runge-Kutta-method  {}'.format(name3(path1)))
	plt.legend()
	plt.show()
	#plt.savefig()


# plot3(parser(PATH), PATH)
"""
Shape = 3 * 100
Name = 'EulerAbove1000_100'
os.mkdir(os.path.join(os.getcwd(), Name))
Data = parser(os.path.join(os.getcwd(), f'{Name}.npy'))
for i in range(len(Data)):
	plot2(Data[i], i)
"""

# plot3(parser(PATH), PATH)
"""
for file in os.scandir():
	print(file)
"""
Path1 = 'EulerBelow1000_50_0_001Energy.npy'
Path2 = 'RungeBelow1000_50_0_001Energy.npy'
plotdiff(Path1, Path2)
