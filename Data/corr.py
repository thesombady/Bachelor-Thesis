import numpy as np
import os, sys

PATH = os.getcwd()
Name = 'EulerBelow1000_100_1.npy'
N = 100
Shape = 3 * N

# FWHM -> gamma_alpha * (n_alpha + 1); alpha in {c,h}


def parser(path) -> np.array:
    with open(path, 'rb') as file:
        data = np.load(file)
    return data


def number(rho):
    particles = 0
    for m in range(3):
        for j in range(N):
            for n in range(3):
                for l in range(N):
                    val = rho[j, m][l, n]
                    if val != 0:
                        particles += 1
    return particles
