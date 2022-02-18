import os
import numpy as np
import matplotlib.pyplot as plt
N = 50
Delta = 0.001
os.chdir('..')
os.chdir('newv')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}


def parser(path):
	with open(path, 'rb') as file:
		return np.load(file)


def plot(path):
	data = parser(path)
	fig, ax = plt.subplots(1, 1, figsize=(7, 5))
	xarray = np.array([Delta * i for i in range(len(data))])
	ax.plot(xarray, data.real, '-', markersize=1, label=r'$\mathfrak{Re}[\langle\hat{a}\rangle]$')
	ax.plot(xarray, data.imag, '-', markersize=1, label=r'$\mathfrak{Im}[\langle\hat{a}\rangle]$')
	# ax.plot(xarray, data.imag, '-', markersize=1, label=r'$\mathfrak{Im}[\langle\hat{a}\rangle]$')
	ax.set_xlabel(r'Time, $\gamma_h^{-1}$', size=15, labelpad=-3)
	ax.set_ylabel(r'$\langle \hat{a}\rangle$', size=15)
	plt.title('Mean field when operating\nabove masing threshold', size=17)
	plt.grid()
	plt.legend(loc=4, fontsize=14)
	plt.yticks([-5, 0, 5], size=17)
	plt.xticks([0, 1, 2], size=17)
	# plt.show()
	plt.savefig('CoherentMeanAbove.pdf')


Path = 'MeanAboveCoherent.npy'
plot(Path)