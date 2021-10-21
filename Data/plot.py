
import os
import numpy as np
import matplotlib.pyplot as plt

# PATH = "EulerAbove1000_100_2Energy.npy"
deltas = 0.0001


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
		return 'w_f = 37.5'
	elif "Above" in path:
		return 'w_f = 150'
	elif "Below" in path:
		return 'w_f = 34'
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


"""
Shape = 3 * 100
Name = 'EulerAbove1000_100'
os.mkdir(os.path.join(os.getcwd(), Name))
Data = parser(os.path.join(os.getcwd(), f'{Name}.npy'))
for i in range(len(Data)):
	plot2(Data[i], i)
"""

# plot3(parser(PATH), PATH)
for file in os.scandir():
	print(file)
