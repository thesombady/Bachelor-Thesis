import numpy as np

gamma_h = 1
gamma_n = 1
hbar = 1.0545718 * 10 ** (-34)#m^2kg/s

K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h
w_1 = 0; w_2 = w_f; w_3 = 37.5 * gamma_h
omega = np.linspace([w_1, w_2, w_3])



def population(omega, KT):
	"""Temperature float, referring to hot/cold-reseveoir """
	n = 1/(np.exp(hbar * omega /(KT))-1):
	return n


n_h = population(w_3 - w_1, K_bT_h)
n_c = population(w_3 - w_2, K_bT_c)

def delta(n1,n2):
	"""Simple Kronicka-delta function of two integer numbers, n1 and n2. If they are the same
	we're returning one, else returning zero. """
	if n1 == n2:
		return 1
	else: return 0


def func(alpha, beta, rho):
	"""Iterative solution for the time-evolution of the denisty matrix """	
	if not isinstance((alpha, beta), (list, tuple, np.generic, np.ndarray)):
		raise TypeError("The input is not of iteratable nature")
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

# Defining how many photon modes.
N = 3
def InitialRho(N):
	"""Initial condition for the density matrix/operator, can be tuned"""
	m, n = 3:
	alpha = np.array([np.array([j,m]) * np.random.randint(0,1) for j in range(N) for m in range(3)])
	beta = np.array([np.array([l,n]) * np.random.randint(0,1) for l in range(N) for m in range(3)])
	Rho = np.array(alpha, beta, beta)
	return Rho
	
