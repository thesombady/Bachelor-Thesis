import numpy as np
import matplotlib.pyplot as plt
import os

N = 3


def parser(path):
    """Standard parser that imports a .npy file.
    Does not parse the content but rather the file."""
    with open(path, 'rb') as file:
        return np.load(file)

