import numpy as np  # Used for computation.
# from Tensor import *
import time
import sys
import logging
import os
global NAME
np.set_printoptions(precision=3, suppress=True, threshold=81)
# Indicate, Run-Photons
NAME = 'Above1000_100'
logging.basicConfig(filename=os.path.join(os.getcwd(), f'Log/{NAME}.csv'), encoding='utf-8', level=logging.DEBUG)
# Increasing recursive limit
sys.setrecursionlimit(2000)
"""
Initial conditions for the program.
Can be changed, especially the reservoir settings as well as the number of particles N.
"""


time1 = time.time()
global N, n_h, n_c, deltas
N = 100  # Number of particles.
gamma_h = 1
gamma_c = 1
hbar = 1  # 1.0545718 * 10 ** (-34)#m^2kg/s
deltas = 0.0001

# Above lasing threshold

K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h  # Lasing angular frequency
# w_1 = 0; w_2 = w_f; w_3 = 150 * gamma_h  # Above lasing threshold
w_0 = 0
w_1 = w_f
w_2 = 150 * gamma_h
omega = np.array([w_0, w_1, w_2])  # An array of the "energies" of the levels
logging.info(f"The computation is done for {NAME}, with the following settings"
		f"K_bT_c = {K_bT_c}, K_bT_h = {K_bT_h}, gamma_h = {gamma_h}, gamma_c = {gamma_c}"
		f"g = {g}, w_f = {w_f}, omegas = {omega}. Deltas =  {deltas}")


def population(w, kt) -> float:
	"""Temperature float, referring to hot/cold-reservoir """
	n = 1/(np.exp(hbar * w / kt) - 1)
	return n


n_h = population(w_2 - w_0, K_bT_h)
n_c = population(w_2 - w_1, K_bT_c)


def delta(n_1, n_2) -> int:
	if n_1 == n_2:
		return 1
	else:
		return 0


def rhodot(alpha, beta, rho) -> complex:
	"""Iterative solution for the time-evolution of the density matrix.
	The solution is derived from Lindblad's master equation, with reference of Niduenze notation,
	and the correspond system."""
	if not isinstance((alpha, beta), (list, tuple, np.generic, np.ndarray)):
		raise TypeError("The input is not of iterable nature")
	j, m = alpha[0], alpha[1]
	l, n = beta[0], beta[1]

	def maximum(p, s, k, d) -> complex:
		"""Returning the null-state a^\dagger|a> = 0 * |null> if a is the boundary."""
		if p == N or k == N:
			return 0
		else:
			return rho[p, s][k, d]

	def minimum(p, s, k, d) -> complex:
		"""Returning the null-state a|0> = 0 * |null>."""
		if p == -1 or k == -1:
			return 0
		else:
			return rho[p, s][k, d]

	# Check w_f term, might be wrong, also check g factor, since it's dealing with evolution
	"""
	var = (1 / 1j * (w_1 * (delta(m, 1 - 1) * rho[j, 1 - 1][l, n] - delta(n, 1 - 1) * rho[j, m][l, 1 - 1])
		+ w_2 * (delta(m, 2 - 1) * rho[j, 2 - 1][l, n] - delta(n, 2 - 1) * rho[j, m][l, 2 - 1])
		+ w_3 * (delta(m, 3 - 1) * rho[j, 3 - 1][l, n] - delta(n, 3 - 1) * rho[j, m][l, 3 - 1])
		+ w_f * rho[j, m][l, n] * (j - l)
		+ g * (np.sqrt(j) * delta(m, 1 - 1) * minimum(j - 1, 2 - 1, l, n)
		+ np.sqrt(j + 1) * delta(m, 2 - 1) * maximum(j + 1, 1 - 1, l, n)
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
	"""
	var = (1/1j*(w_0 * (delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])  # first
		+ w_1 * (delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1])  # second
		+ w_2 * (delta(m, 2) * rho[j, 2][l, n] - delta(n, 1) * rho[j, m][l, 2])  # third
		+ w_f * rho[j, m][l, n] * (j - l)  # lasing
		+ g * (np.sqrt(j) * delta(m, 0) * minimum(j - 1, 1, l, n)
		+ np.sqrt(j + 1) * delta(m, 1) * maximum(j + 1, 0, l, n)
		- np.sqrt(l + 1) * delta(n, 1) * maximum(j, m, l + 1, 0)
		- np.sqrt(l) * delta(n, 0) * minimum(j, m, l - 1, 1)))  # Jaynes-Cumming
		+ gamma_h * (n_h + 1) * (2 * delta(m, 0) * delta(n, 0) * rho[j, 2][l, 2]  # Liouvillian terms
		- delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
		+ gamma_h * n_h * (2 * delta(m, 2) * delta(n, 2) * rho[j, 0][l, 0]
		- delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])
		+ gamma_c * (n_c + 1) * (2 * delta(m, 1) * delta(n, 1) * rho[j, 2][l, 2]
		- delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
		+ gamma_c * n_c * (2 * delta(m, 2) * delta(n, 2) * rho[j, 1][l, 1]
		- delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1]))
	"""
	var = (1 / 1j * (w_0 * (delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])  # first
			+ w_1 * (delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1])  # second
			+ w_2 * (delta(m, 2) * rho[j, 2][l, n] - delta(n, 1) * rho[j, m][l, 2])  # third
			+ w_f * rho[j, m][l, n] * (j - l))  # lasing
			+ g * 2 * (np.sqrt(j) * delta(m,0) * minimum(j -1, 1, l, n)
			+ np.sqrt(j + 1) * delta(n, 1) * maximum(j + 1, 0, l, n)).imag# Jaynes-Cumming
			+ gamma_h * (n_h + 1)  * (2 * delta(m, 0) * delta(n, 0) * rho[j, 2][l, 2]  # Liouvillian terms
			- delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
			+ gamma_h * n_h * (2 * delta(m, 2) * delta(n, 2) * rho[j, 0][l, 0]
			- delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])
			+ gamma_c * (n_c + 1) * (2 * delta(m, 1) * delta(n, 1) * rho[j, 2][l, 2]
			- delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
			+ gamma_c * n_c * (2 * delta(m, 2) * delta(n, 2) * rho[j, 1][l, 1]
			- delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1]))
	return var


def initialrho(n: int) -> np.array:
	"""Returns a initial-condition density matrix."""
	ten = np.full((n, 3, n, 3), 0, dtype=complex)
	for j in range(n):
		for m in range(3):
			for l in range(n):
				for k in range(3):
					if m == 0 and k == 0 and j == 1 and l == 1:
						ten[j, m][l, k] = 1
	return ten/ten.sum()  # Normalizing


def zerorho(n) -> np.array:
	"""Returns a tensor of rank(4) with dimension (3N)^2."""
	ten = np.full((n, 3, n, 3), 0, dtype=complex)
	return ten


def helper(rho) -> np.array:
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
	k1 = helper(rho)  # computes rhodot
	rho1 = rho + k1 * deltas
	print(rho.reshape(3 * N, -1).trace())
	if n > 0:
		Iterations.append(rho1)
		logging.info(f'Iteration {n} yields : {rho1}\nTrace is:{rho1.reshape(3 * N, - 1).trace()} for iteration {n}')
		euler(rho1, n - 1)
	else:
		path = os.path.join(os.getcwd(), f'Euler{NAME}.npy')  # "Euler{name}.npy".format(name = NAME))
		with open(path, 'wb') as file:
			np.save(file, np.array(Iterations))


def runge(rho, n):
	logging.info(f"Using Runge-Kutta method")
	if not isinstance(rho, (np.ndarray, np.generic)):
		raise TypeError("Input is of wrong format")
	if n > 1:
		k_1 = helper(rho)
		k_2 = helper(rho + deltas / 2 * k_1)
		k_3 = helper(rho + deltas / 2 * k_2)
		k_4 = helper(rho + deltas * k_3)
		rho1 = rho + (k_1 + 2 * k_2 + 2 * k_3 + k_4)
		# rho1 = rho1/np.sqrt(rho1.real.sum() ** 2 + rho1.imag.sum() ** 2)
		Iterations.append(rho1)
		logging.info(f'Iteration {n} yields : {rho1}')
		runge(rho1, n - 1)
	else:
		path = os.path.join(os.getcwd(), f'Runge{NAME}.npy')
		with open(path, 'wb') as file:
			np.save(file, np.array(Iterations))


def qfunction(energy, power) -> float:
	"""Computes the q-factor for a given system"""
	# Note, we need first to compute E = tr(rho * H), and dE/dt = P to compute the q-factor
	q = 2 * np.pi * w_f * energy / power
	return q


Rho0 = initialrho(n=N)
# print(Rho0.reshape(N * 3, - 1))
Iterations.append(Rho0)
euler(Rho0, 1000)
"""
for i in range(len(Iterations)):
	print(Iterations[i].reshape(3 * N, -1).trace())
# print(Iterations[-1].reshape(3*N, -1))  #This works to reshape to a 3*N matrix.
"""
