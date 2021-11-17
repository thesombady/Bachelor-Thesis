
import os
import numpy as np
import matplotlib.pyplot as plt

PATH = "RungeAbove1000_100_2Energy.npy"
deltas = 0.01
N = 3
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
	ax = plt.axes()
	values1 = parser(path1)
	values2 = parser(path2)
	assert len(values1) == len(values2), 'Different lengths'
	xlist = np.array([i for i in range(len(values1))]) * deltas
	ax.plot(xlist, values1, '.', markersize=0.8, color='red', label="Euler")
	ax.plot(xlist, values2, '.', markersize=0.8, color='blue', label="Runge-Kutta")
	"""
	ax2 = plt.axes([0.5, 0.3, 0.3, 0.3])
	ax2.plot(xlist, values1, '.', markersize=0.8, color='red', label="Euler")
	ax2.plot(xlist, values2, '.', markersize=0.8, color='blue', label="Runge-Kutta")
	ax2.set_ylim([6, 10])
	ax2.set_xlim([0, 1])
	"""
	ax.set_xlabel(r'Time, a.u')
	ax.set_ylabel(r'Energy, a.u')
	ax.set_title('Average energy comparison of\nEuler-, and Runge-Kutta-method  {}'.format(name3(path1)))
	plt.legend()
	plt.show()
	#plt.savefig()


def plote(data1, data2, data3, data4, data5, data6, data7, data8):
	fig, ax = plt.subplots(2, 1, figsize=(8, 7))
	xlist1 = np.array([i * 0.02 for i in range(len(data1))])
	xlist2 = np.array([i * 0.01 for i in range(len(data3))])
	xlist3 = np.array([i * 0.005 for i in range(len(data5))])
	xlist4 = np.array([i * 0.001 for i in range(len(data7))])
	fig.tight_layout(pad=3.0)
	#ax[0].plot(xlist1, data1.real, '--', markersize=1, label=r'Euler, $\Delta t = 0.02$', color='blue')
	ax[0].plot(xlist1, data2.real, '-', markersize=1, label=r'Euler, $\Delta t = 0.02$', color='lightskyblue')
	# ax[0].plot(xlist2, data3.real, '--', markersize=1, label=r'Runge, $\Delta t = 0.01$', color='red')
	ax[0].plot(xlist2, data4.real, '-', markersize=1, label=r'Euler, $\Delta t = 0.01$', color='coral')
	ax[0].plot(xlist3, data5.real, '--', markersize=1, label=r'Runge, $\Delta t = 0.005$', color='green')
	ax[0].plot(xlist3, data6.real, '-', markersize=1, label=r'Euler, $\Delta t = 0.005$', color='limegreen')
	ax[0].plot(xlist4, data7.real, '--', markersize=1, label=r'Runge $\Delta t = 0.001$', color='darkgrey')
	ax[0].plot(xlist4, data8.real, '-', markersize=1, label=r'Euler $\Delta t = 0.001$', color='gray')
	# ax[1].plot(xlist1, data1.real, '--', markersize=1, label=r'Runge, $\Delta t = 0.02$', color='blue')
	ax[1].plot(xlist1, data2.real, '-', markersize=1, label=r'Euler, $\Delta t = 0.02$', color='lightskyblue')
	# ax[1].plot(xlist2, data3.real, '--', markersize=1, label=r'Runge, $\Delta t = 0.01$', color='red')
	# ax[1].plot(xlist2, data4.real, '-', markersize=1, label=r'Euler, $\Delta t = 0.01$', color='coral')
	ax[1].plot(xlist3, data5.real, '--', markersize=1, label=r'Runge, $\Delta t = 0.005$', color='green')
	ax[1].plot(xlist3, data6.real, '-', markersize=1, label=r'Euler, $\Delta t = 0.005$', color='limegreen')
	ax[1].plot(xlist4, data7.real, '--', markersize=1, label=r'Runge $\Delta t = 0.001$', color='darkgrey')
	ax[1].plot(xlist4, data8.real, '-', markersize=1, label=r'Euler $\Delta t = 0.001$', color='gray')
	ax[0].set_ylim(0, 1)
	ax[1].set_xlim(0, 50)
	ax[1].set_ylim(0, 0.5)
	ax[0].set_title('Comparison of Euler and Runge-Kutta')
	ax[1].set_title('Zoomed on problimatic area')
	ax2 = plt.axes([0.4, 0.175, 0.3, 0.2])
	# ax2.plot(xlist1, data1.real, '--', markersize=1, label=r'Euler, $\Delta t = 0.02$', color='blue')
	ax2.plot(xlist1, data2.real, '-', markersize=1, label=r'Runge, $\Delta t = 0.02$', color='lightskyblue')
	# ax2.plot(xlist2, data3.real, '--', markersize=1, label=r'Euler, $\Delta t = 0.01$', color='red')
	# ax2.plot(xlist2, data4.real, '-', markersize=1, label=r'Runge, $\Delta t = 0.01$', color='coral')
	ax2.plot(xlist3, data5.real, '--', markersize=1, label=r'Euler, $\Delta t = 0.005$', color='green')
	ax2.plot(xlist3, data6.real, '-', markersize=1, label=r'Runge, $\Delta t = 0.005$', color='limegreen')
	ax2.plot(xlist4, data7.real, '--', markersize=1, label=r'Euler $\Delta t = 0.001$', color='darkgrey')
	ax2.plot(xlist4, data8.real, '-', markersize=1, label=r'Runge $\Delta t = 0.001$', color='gray')
	# ax2.legend(loc=4)
	ax2.set_xlim(0, 50)
	ax2.set_ylim(0, 0.01)
	ax2.set_ylabel(r'$\langle E \rangle, ~ \hbar\omega_f$')
	ax2.set_ylabel(r'$\langle E \rangle, ~ \hbar\omega_f$')
	for axes in ax:
		axes.set_xlabel(r'Time, $\gamma_h$s')
		axes.set_ylabel(r'$\langle E \rangle, ~ \hbar\omega_f$')
		axes.legend(loc=4)
	plt.show()
	plt.clf()


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
def name1(path):
	if 'Euler' in path:
		return 'Euler'
	else:
		return 'Runge'


def plots(path, delta):
	data = parser(path).real
	xlist = np.array([delta * i for i in range(len(data))])
	plt.plot(xlist, data, '-', markersize=1, label=f'{name1(path)} {delta}')
	plt.legend()
	plt.show()


os.chdir('..')
Path1 = 'EulerAbove5000_3_0.02_CFalseEnergy.npy'
Path2 = 'RungeAbove5000_3_0.02_CFalseEnergy.npy'
Path3 = 'EulerAbove10000_3_0.01_CFalseEnergy.npy'
Path4 = 'RungeAbove10000_3_0.01_CFalseEnergy.npy'
Path5 = 'EulerAbove20000_3_0.005_CFalseEnergy.npy'
Path6 = 'RungeAbove20000_3_0.005_CFalseEnergy.npy'
Path7 = 'EulerAbove100000_3_0.001_CFalseEnergy.npy'
Path8 = 'RungeAbove100000_3_0.001_CFalseEnergy.npy'
# plotdiff(Path1, Path2)
"""
plote(parser(Path1), parser(Path2), parser(Path3), parser(Path4), parser(Path5), parser(Path6),
	parser(Path7), parser(Path8))
"""


plots(Path8, 0.001)


