import numpy as np
import matplotlib.pyplot as plt


def parser(path):
	"""A parser function, which utilizes the functionality of numpy.save.
	Using this method, each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		data = np.load(file)
	return data


Path = "/Data/Data_1mode/EulerAbove100_3.npy"
Data = parser(Path)

