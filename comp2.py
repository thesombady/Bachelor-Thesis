import numpy as np  # Used for computation.
import time
import sys
import logging
import os
global NAME
np.set_printoptions(precision=5, suppress=True, threshold=81)
# Indicate, Run-Photons


def parser():
	global w_2
	opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
	argv = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

	assert len(sys.argv) <= 5, 'To many arguments'


	def name(argument):
		if 'Above' in argument:
			w_2 = 150 * gamma_h
		elif 'Below' in argument:
			w_2 = 34 * gamma_h
		elif 'Lasing' in argument:
			w_2 = 37.5 * gamma_h



	if "-h" in opts:
		print("Arg1 = Number of photon modes, Arg2 = Number of iterations, Arg3 = Number of iterations\nArg4 = Mode of operation")
		sys.exit



NAME = 'Below1000_50_1'
KEY = 'Runge'
itera = 1000
logging.basicConfig(filename=os.path.join(os.getcwd(), f'Log/{NAME + KEY}.csv'), encoding='utf-8', level=logging.DEBUG)
# Increasing recursive limit
sys.setrecursionlimit(2000)
"""
Initial conditions for the program.
Can be changed, especially the reservoir settings as well as the number of particles N.
"""
Method = {
	'Euler': lambda rho, n: euler(rho, n),
	'Runge': lambda rho, n: runge(rho, n),
}

global N, n_h, n_c, deltas
deltas = 0.0001
N = 50  # Number of particles.
gamma_h = 1
gamma_c = 1
hbar = 1  # 1.0545718 * 10 ** (-34)#m^2kg/s

# Above lasing threshold

K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h  # Lasing angular frequency
# w_1 = 0; w_2 = w_f; w_3 = 150 * gamma_h  # Above lasing threshold
w_0 = 0
w_1 = w_f
w_2 = 34 * gamma_h  # This is the one we change for laser, 34, 37.5, 150 respectively.
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
		"""Returning the null-state
		a^\dagger|a> = 0 * |null> if a is the boundary."""
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

	var = (1/1j * (w_0 * (delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])  # first
			+ w_1 * (delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1])  # second
			+ w_2 * (delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])  # third
			+ w_f * rho[j, m][l, n] * (j - l))  # lasing
			+ 2 * g * (np.sqrt(j) * delta(m, 0) * minimum(j - 1, 1, l, n)
			+ np.sqrt(j + 1) * delta(m, 1) * maximum(j + 1, 0, l, n)).imag  # Jaynes - Cumming
			+ gamma_h * (n_h + 1) * (2 * delta(n, 0) * delta(m, 0) * rho[j, 2][l, 2]
			- delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
			+ gamma_h * n_h * (2 * delta(m, 2) * delta(n, 2) * rho[j, 0][l, 0]
			- delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])  # Hot - Liouvillian
			+ gamma_c * (n_c + 1) * (2 * delta(m, 1) * delta(n, 1) * rho[j, 2][l, 2]
			- delta(m, 2) * rho[j, 2][l, 2] - delta(n, 2) * rho[j, m][l, 2])
			+ gamma_c * n_c * (2 * delta(m, 2) * delta(n, 2) * rho[j, 1][l, 1]
			- delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1]))  # Cold - Liouvillian
	return var


def initialrho(n: int) -> np.array:
	"""Returns a initial-condition density matrix, a photon in the ground-state of the atom"""
	ten = np.full((n, 3, n, 3), 0, dtype=complex)
	for j in range(n):
		for m in range(3):
			for l in range(n):
				for k in range(3):
					if m == 1 and k == 1 and j == 1 and l == 1:
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
	if n > 0:
		k1 = helper(rho)  # computes rhodot
		rho1 = rho + k1 * deltas
		print(f'Iteration:{n}', rho.reshape(3 * N, - 1).trace())
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
	if n > 0:
		k1 = helper(rho)
		k2 = helper(rho + deltas / 2 * k1)
		k3 = helper(rho + deltas / 2 * k2)
		k4 = helper(rho + deltas * k3)
		rho1 = rho + (k1 + 2 * k2 + 2 * k3 + k4) * deltas /6
		Iterations.append(rho1)
		print(rho.reshape(3 * N, -1).trace(), n)
		logging.info(f'Iteration {n} yields : {rho1}')
		runge(rho1, n - 1)
	else:
		path = os.path.join(os.getcwd(), f'Runge{NAME}.npy')
		with open(path, 'wb') as file:
			np.save(file, np.array(Iterations))


start = time.time()
Rho0 = initialrho(n=N)
# print(Rho0.reshape(N * 3, - 1))
Iterations.append(Rho0)
# euler(Rho0, itera)
try:
	Method[KEY](Rho0, itera)
except:
	pass
logging.info(f'Runtime {time.time()-start}')
"""

for i in range(len(Iterations)):
	print(Iterations[i].reshape(3 * N, -1).trace())
# print(Iterations[-1].reshape(3*N, -1))  #This works to reshape to a 3*N matrix.
"""