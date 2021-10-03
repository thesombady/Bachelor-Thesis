
import os
import numpy as np
import matplotlib.pyplot as plt
def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save. Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		Data = np.load(file)
	return Data

def plot(data):
	"""Data has to be of a matrix"""
	plt.plot(data)
	plt.show()

path = os.path.join("/Users/andreasevensen/Documents/GitHub/Bachelor-Thesis", 'Data')
path1 = os.path.join(path, 'test.npy')
print(parser(path1)[-1])

