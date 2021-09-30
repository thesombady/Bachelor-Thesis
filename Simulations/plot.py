import numpy as np
import matplotlib.pyplot as plt


def parser(path):
	try:
		with open(path) as f:
			val = f.read()
	except:
		raise Exception("Could not read data")
	return val


