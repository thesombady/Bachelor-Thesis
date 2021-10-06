import numpy as np
import matplotlib.pyplot as plt
import os
Name = 'EulerAbove100_3.npy'
Path = os.path.join('/Data', Name)


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)

print(parser(Path)[1].reshape(3 * 3, -1))
print(parser(Path)[2].reshape(3 * 3, -1))

