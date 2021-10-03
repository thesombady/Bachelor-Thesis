import numpy as np#Used for computation.
#import pandas as pd#Used for saving the computed data.
from Tensor import *
import time
import os

import sys
#Increasing recursive limit
sys.setrecursionlimit(2000)
"""
Initial conditions for the program.
Can be changed, especially the resevoir settings as well as the number of particles N.
"""
time1 = time.time()
global N, n_h, n_c, deltas
N = 3 #Number of particles.
gamma_h = 0.001
gamma_c = 0.001
hbar = 1.0545718 * 10 ** (-34)#m^2kg/s
deltas = 10 ** (-9)

K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h#Lasing angular frequency
w_1 = 0; w_2 = w_f; w_3 = 37.5 * gamma_h
omega = np.array([w_1, w_2, w_3])#An array of the "energies" of the levels



def population(omega, KT):
	"""Temperature float, referring to hot/cold-reseveoir """
	n = 1/(np.exp(hbar * omega /(KT))-1)
	return n

n_h = population(w_3 - w_1, K_bT_h)
n_c = population(w_3 - w_2, K_bT_c)


def delta(n1,n2):
	"""Simple Kronicka-delta function of two integer numbers, n1 and n2. If they are the same
	we're returning one, else returning zero. """
	if n1 == n2:
		return 1
	else: return 0

def rhodot(alpha, beta, rho):
	"""Iterative solution for the time-evolution of the denisty matrix.
    The solution is derived from Lindblads master equation, with reference of Niduenze nomenclentur,
    and the correspond system."""
	if not isinstance((alpha, beta), (list, tuple, np.generic, np.ndarray)):
		raise TypeError("The input is not of iterable nature")
	j,m = alpha[0], alpha[1]
	l,n = beta[0], beta[1]
	var = ( 1 / 1j * (w_1 * (delta(m,1)*rho[[j,1],[l,n]]- delta(n,1)*rho[[j,m],[l,1]])
		+ w_2 * (delta(m,2) * rho[[j,2],[l,n]] - delta(n,2) * rho[[j,m],[l,2]])
		+ w_3 * (delta(m,3) * rho[[j,m],[l,3]] - delta(n,3) * rho[[j,m],[l,3]])
		+ w_f * rho[[j,m], [l,n]] * (j-l)
		+ g * (np.sqrt(j) * delta(m,1) * rho[[j-1,2],[l,n]] + np.sqrt(j+1) * delta(m,2) * rho[[j+1,m],[l,n]]
		- np.sqrt(l+1) * delta(n,2) * rho[[j,m],[l+1,1]] -np.sqrt(l) * delta(n,1) * rho[[j,m],[l-1,2]]))
		+ gamma_h * (n_h + 1) * (2 * delta(m,1) * delta(n,1) * rho[[j,3], [l,3]]
		- delta(m,3) * rho[[j,3],[n,l]] - delta(n,3) * rho[[j,m],[l,3]])
		+ gamma_h * n_h * (2 * delta(m,3) * delta(n,3) * rho[[j,1],[l,1]]
		- delta(m,1) * rho[[j,1],[l,n]] - delta(n,1) * rho[[j,m],[l,1]])
		+ gamma_c * (n_c + 1) * (2 * delta(m,2) * delta(n,2) * rho[[j,3],[l,3]]
		- delta(m,3) * rho[[j,3],[l,n]] - delta(n,3) * rho[[j,m],[l,3]])
		+ gamma_c * n_c * (2 * delta(m,3) * delta(n,3) * rho[[j,2],[l,2]]
		- delta(m,2) * rho[[j,2],[l,n]] - delta(n,2) * rho[[j,m],[l,2]]))
	return var

# Defining how many photon mode.

#Inital Density matrix
def InitialRho():
	"N is the number of photon modes, maximum of 100, which means 0-99 photon-modes."
	Rho0 = tens(N = N).zerostates()
	return Rho0

def Helper(Rho0):
	"""Helper function, which computes rho-dot, for a given density operator rho. Is used in Runge function, to iterate either with euler,
	Runge-Kutta method, in order to solve a first order differential equation at time t."""
	Rho = tens(N = N).set0()
	for m in range(1, 4):
		for j in range(N):
			for n in range(1, 4):
				for l in range(N):
					var = rhodot([j,m],[l,n], Rho0)
					#print(var, f"j,m = {j,m - 1}", f"l,n {l,n - 1}")
					Rho._set(Val = var, index = [[j,m], [l,n]])
	#print(Rho)
	#Rho._save('Test')
	return Rho


Iterations = []
def Runge(Rho, n):
	"""Iterative first order differential equation solver, utalizes either Euler method, or Runge-Kutta method
	in order to solve the first order differential equation system. Since this is iterative, with increments fixed increment,
	one can easily focus on the time-evolution."""
	if n > 0:
		k_1 = Helper(Rho)
		#k_2 = Helper(Rho + deltas/2 * k_1)
		#k_3 = Helper(Rho + deltas/2 * k_2)
		#k_4 = Helper(Rho + deltas * k_3)
		#yn_1 = Rho + 1/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
		yn_1 = Rho + deltas * k_1
		yn_1 = yn_1 / yn_1.Data.sum()#Normalize
		Iterations.append(yn_1.Data)
		Runge(yn_1, n-1)
	else:
		print("Last entry",Rho)
		path = os.path.join(os.getcwd(), "test.npy")
		with open(path, 'wb') as f:
			np.save(f, np.array(Iterations))
Runge(InitialRho(), 10)
print(time.time()-time1)