import numpy as np  # Used for computation.
# from Tensor import *
import time
import sys
import logging
import os
global NAME
np.set_printoptions(precision=2, suppress=True, threshold=81)
# Indicate, Run-Photons
NAME = 'Lasing100_3'
logging.basicConfig(filename=f'{NAME}.csv', encoding='utf-8', level=logging.DEBUG)
# Increasing recursive limit
sys.setrecursionlimit(2000)
"""
Initial conditions for the program.
Can be changed, especially the reservoir settings as well as the number of particles N.
"""
time1 = time.time()
global N, n_h, n_c, deltas
N = 3  # Number of particles.
gamma_h = 1
gamma_c = 1
hbar = 1  # 1.0545718 * 10 ** (-34)#m^2kg/s
deltas = 10 ** (-9)
"""Above lasing threshold"""
K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h  # Lasing angular frequency
w_1 = 0; w_2 = w_f; w_3 = 37.5 * gamma_h  # Above lasing threshold
omega = np.array([w_1, w_2, w_3])  # An array of the "energies" of the levels
logging.info(f"The computation is done for {NAME}, with the following settings"
		f"K_bT_c = {K_bT_c}, K_bT_h = {K_bT_h}, gamma_h = {gamma_h}, gamma_c = {gamma_c}"
		f"g = {g}, w_f = {w_f}, omegas = {omega}. Deltas =  {deltas}")


def population(w, kt):
	"""Temperature float, referring to hot/cold-reservoir """
	n = 1/(np.exp(hbar * w / kt) - 1)
	return n


n_h = population(w_3 - w_1, K_bT_h)
n_c = population(w_3 - w_2, K_bT_c)


"""
tensor = np.full((3,3,3,3),0)
print(tensor)
print("--------------")
tensor[0,0][0,1] = 1
print(tensor)
"""


def delta(n_1, n_2):
	if n_1 == n_2:
		return 1
	else:
		return 0


def rhodot(alpha, beta, rho):
	"""Iterative solution for the time-evolution of the density matrix.
	The solution is derived from Lindblad's master equation, with reference of Niduenze notation,
	and the correspond system."""
	if not isinstance((alpha, beta), (list, tuple, np.generic, np.ndarray)):
		raise TypeError("The input is not of iterable nature")
	j, m = alpha[0], alpha[1]
	l, n = beta[0], beta[1]

	def maximum(j, m, l, n):
		if j == N or l == N:
			return 0
		else:
			return rho[j, m][l, n]

	def minimum(j, m, l, n):
		if j == -1 or l == -1:
			return 0
		else:
			return rho[j, m][l, n]

	# Check w_f term, might be wrong, also check g factor, since it's dealing with evolution
	var = (1 / 1j * (w_1 * (delta(m, 1 - 1) * rho[j, 1 - 1][l, n] - delta(n, 1 - 1) * rho[j, m][l, 1 - 1])
		+ w_2 * (delta(m, 2 - 1) * rho[j, 2 - 1][l, n] - delta(n, 2 - 1) * rho[j, m][l, 2 - 1])
		+ w_3 * (delta(m, 3 - 1) * rho[j, 3 - 1][l, n] - delta(n, 3 - 1) * rho[j, m][l, 3 - 1])
		+ w_f * rho[j, m][l, n] * (j - l)
		+ g * (np.sqrt(j) * delta(m, 1 - 1) * minimum(j - 1, 2 - 1, l, n)
		+ np.sqrt(j + 1) * delta(m, 2 - 1) * maximum(j + 1, m, l, n)
		- np.sqrt(l + 1) * delta(n, 2 - 1) * maximum(j, m, l + 1, 1 - 1)
		- np.sqrt(l) * delta(n, 1 - 1) * minimum(j, m, l - 1, 2 - 1)))
		+ gamma_h * (n_h + 1) * (2 * delta(m, 1 - 1) * delta(n, 1 - 1) * rho[j, 3 - 1][l, 3 - 1]
		- delta(m, 3 - 1) * rho[j, 3 - 1][l, n] - delta(n, 3 - 1) * rho[j, m][l, 3 - 1])
		+ gamma_h * n_h * (2 * delta(m, 3 - 1) * delta(n, 3 - 1) * rho[j, 1 - 1][l, 1 - 1]
		- delta(m, 1 - 1) * rho[j, 1 - 1][l, n] - delta(n, 1 - 1) * rho[j, m][l, 1 - 1])
		+ gamma_c * (n_c + 1) * (2 * delta(m, 2 - 1) * delta(n, 2 - 1) * rho[j, 3 - 1][l, 3 - 1]
		- delta(m, 3 - 1) * rho[j, 3 - 1][l, n] - delta(n, 3 - 1) * rho[j, m][l, 3 - 1])
		+ gamma_c * n_c * (2 * delta(m, 3 - 1) * delta(n, 3 - 1) * rho[j, 2 - 1][l, 2 - 1]
		- delta(m, 2 - 1) * rho[j, 2 - 1][l, n] - delta(n, 2 - 1) * rho[j, m][l, 2 - 1]))
	"""
	var = (1 / 1j * (w_1 * (delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1])
		+ w_2 * (delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
		+ w_3 * (delta(m, 3) * rho[j, 3][l, n] - delta(n, 3) * rho[j, m][l, 3])
		+ w_f * rho[j, m][l, n] * (j - l)
		+ g * (np.sqrt(j) * delta(m, 1) * rho[j - 1, 2][l, n] + np.sqrt(j + 1) * delta(m, 2) * rho[j + 1, m][l, n]
		- np.sqrt(l + 1) * delta(n, 2) * rho[j, m][l + 1, 1] - np.sqrt(l) * delta(n, 1) * rho[j, m][l - 1, 2]))
		+ gamma_h * (n_h + 1) * (2 * delta(m, 1) * delta(n, 1) * rho[j, 3][l, 3]
		- delta(m, 3) * rho[j, 3][l, n] - delta(n, 3) * rho[j, m][l, 3])
		+ gamma_h * n_h * (2 * delta(m, 3) * delta(n, 3) * rho[j, 1][l, 1]
		- delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1])
		+ gamma_c * (n_c + 1) * (2 * delta(m, 2) * delta(n, 2) * rho[j, 3][l, 3]
		- delta(m, 3) * rho[j, 3][l, n] - delta(n, 3) * rho[j, m][l, 3])
		+ gamma_c * n_c * (2 * delta(m, 3) * delta(n, 3) * rho[j, 2][l, 2]
		- delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2]))
	"""
	return var


def initialrho(n=3):
	ten = np.full((n, 3, n, 3), 0, dtype=complex)
	for m in range(3):
		for j in range(n):
			for n in range(3):
				for l in range(n):
					if m == 0 and n == 0:
						ten[j, m][l, n] = np.random.randint(1, 10) * 0.1
	return ten


def zerorho(n=3):
	ten = np.full((n, 3, n, 3), 0, dtype=complex)
	return ten


def helper(rho):
	"""Helper function, which computes rho-dot, for a given density operator rho. Is used in Runge function,
	to iterate either with euler,
	Runge-Kutta method, in order to solve a first order differential equation at time t."""
	rho1 = zerorho(n=N)
	for m in range(3):
		for j in range(N):
			for n in range(3):
				for l in range(N):
					var = rhodot([j, m], [l, n], rho)
					rho1[j, m][l, n] = var
	return rho1


Iterations = []


def euler(rho, n):
	logging.info(f"Using Euler method")
	k1 = helper(rho)
	rho1 = rho + k1 * deltas
	if n > 0:
		Iterations.append(rho1)
		logging.info(f'Iteration {n} yields : {rho1}')
		euler(rho1, n - 1)
	else:
		path = os.path.join(os.getcwd(), f'Euler{NAME}.npy')  # "Euler{name}.npy".format(name = NAME))
		with open(path, 'wb') as file:
			np.save(file, np.array(Iterations))


def runge(rho, n):
	logging.info(f"Using Runge-Kutta method")
	if n > 1:
		k_1 = helper(rho)
		k_2 = helper(rho + deltas / 2 * k_1)
		k_3 = helper(rho + deltas / 2 * k_2)
		k_4 = helper(rho + deltas * k_3)
		rho1 = rho + 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
		Iterations.append(rho1)
		logging.info(f'Iteration {n} yields : {rho1}')
		runge(rho1, n - 1)
	else:
		path = os.path.join(os.getcwd(), f'Runge{NAME}.npy')
		with open(path, 'wb') as file:
			np.save(file, np.array(Iterations))


Rho0 = initialrho(n=N)
# print(Rho0.reshape(N * 3, - 1))
Iterations.append(Rho0)
euler(Rho0, 100)
# print(a.reshape(N*3, c-1))  #This works to reshape to a 3*N matrix.
