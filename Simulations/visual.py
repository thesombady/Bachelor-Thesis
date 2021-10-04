import numpy as np
import matplotlib.pyplot as plt


def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save. Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		Data = np.load(file)
	return Data


path = '/Users/andreasevensen/Documents/GitHub/Bachelor-Thesis/Data/EulerLasing100_3.npy'
Data = parser(path)
for i in range(len(Data)):
	print(Data[i].reshape(-1, 3 * 3))
	print(i)
	if i == 5:
		break