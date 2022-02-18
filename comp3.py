import numpy as np  # Used for computation.
import sys
import os
np.set_printoptions(precision=5, suppress=True, threshold=81)

itera = 2000  # Number of iterations
N = 50  # Number of particles.
Iterstep = 1000  # Saving parameter
os.chdir('newv')


def parser():
    """Parsing the commandline arguments"""
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    argv = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    assert len(argv) <= 5, 'To many arguments'

    def name(argument):
        if 'Above' in argument:
            return 150, 'Above'
        elif 'Below' in argument:
            return 34, 'Below'
        elif 'Lasing' in argument:
            return 37.5, 'Lasing'

    def compressor(argument):
        if argument == 'True':
            return True
        else:
            return False

    if "-h" in opts:
        print("""Arg1 = Method to use, Euler & Runge,Arg2 = Delta t(0.001),
        Arg3 = Mode of operation(Above, Below, Lasing), Arg4 = Compressed {True or False}""")
        sys.exit()
    try:
        KEY = str(argv[0])
        deltas = float(argv[1])
        w_2 = name(argv[2])[0]
        NAME = name(argv[2])[1]
        try:
            KEY2 = compressor(argv[3])
        except:
            KEY2 = False
        return KEY, deltas, NAME, w_2, KEY2
    except Exception as e:
        raise e


KEY, deltas, NAME, w_2, KEY2 = parser()


Method = {
    'Euler': lambda rho, n: euler2(rho, n),
    'Runge': lambda rho, n: runge2(rho, n),
}

gamma_h = 1  # usually in units of 10^(-12) seconds
gamma_c = 1  # usually in units of 10^(-12) seconds
hbar = 6.58 * 10 ** (-6)  # In units of eV/ns and the conversion of gamma_h  # 1.0545718 * 10 ** (-34)#m^2kg/s

K_bT_c = 20 * hbar * gamma_h
K_bT_h = 100 * hbar * gamma_h
g = 5 * gamma_h
w_f = 30 * gamma_h  # Lasing angular frequency
w_0 = 0
w_1 = w_f
w_2 = w_2 * gamma_h  # This is the one we change for laser, 34, 37.5, 150 respectively.
omega = np.array([w_0, w_1, w_2])  # An array of the "energies" of the levels


def population(w, kt) -> float:
    """Temperature float, referring to hot/cold-reservoir """
    n = 1/(np.exp(hbar * w / kt) - 1)
    return n


n_h = population(w_2 - w_0, K_bT_h)
n_c = population(w_2 - w_1, K_bT_c)


def delta(n_1, n_2) -> int:
    """A delta function between to integers, returns one if equal, otherwise returns zero."""
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

    var = (
        - 1j * (w_0 * (delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0])
               + w_1 * (delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1])
               + w_2 * (delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2])
               + w_f * rho[j, m][l, n] * (j - l)
               + g * (np.sqrt(j) * delta(m, 0) * minimum(j - 1, 1, l, n)
                    + np.sqrt(j + 1) * delta(m, 1) * maximum(j + 1, 0, l, n)
                    - np.sqrt(l) * delta(n, 0) * minimum(j, m, l - 1, 1)
                    - np.sqrt(l + 1) * delta(n, 1) * maximum(j, m, l + 1, 0)))
        + gamma_h * (n_h + 1) * (
        2 * delta(n, 0) * delta(m, 0) * rho[j, 2][l, 2]
        - delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2]
        )
        + gamma_h * n_h * (
        2 * delta(m, 2) * delta(n, 2) * rho[j, 0][l, 0]
        - delta(m, 0) * rho[j, 0][l, n] - delta(n, 0) * rho[j, m][l, 0]
        )
        + gamma_c * (n_c + 1) * (
        2 * delta(m, 1) * delta(n, 1) * rho[j, 2][l, 2]
        - delta(m, 2) * rho[j, 2][l, n] - delta(n, 2) * rho[j, m][l, 2]
        )
        + gamma_c * n_c * (
        2 * delta(m, 2) * delta(n, 2) * rho[j, 1][l, 1]
        - delta(m, 1) * rho[j, 1][l, n] - delta(n, 1) * rho[j, m][l, 1]
        )
    )
    return var


def initialrho(n: int) -> np.array:
    """Returns a initial-condition density matrix, no photon in the ground-state of the atom"""
    ten = np.full((n, 3, n, 3), 0, dtype=complex)
    """
    for j in range(n):
        for m in range(3):
            for l in range(n):
                for k in range(3):
                    if m == 0 and k == 0 and j == 0 and l == 0:
                        ten[j, m][l, k] = 1
    """
    ten[0, 0][0, 0] += 1
    return ten/ten.sum()  # Normalizing


def initialrho2(n: int) -> np.array:
    """Returns an initial-conditions density operator, no photon in the ground-state"""
    ten = np.full((n, 3, n, 3), 0, dtype=complex)
    al = 1.25
    bl = 1.25
    for j in range(N):
        for l in range(N):
            ten[j, 0][l, 0] += 1 / (al ** l * bl ** j * np.exp(-1/2 * (np.abs(al) ** 2 - np.abs(bl) ** 2)) /
                                    (np.sqrt(np.math.factorial(float(l)) * np.math.factorial(float(j)))))
    print('initial', ten.reshape(3 * N, -1, order='F'))
    return ten / ten.reshape(3 * N, - 1).trace()


def initialrho3(n: int) -> np.array:
    ten = np.full((n, 3, n, 3), 0, dtype=complex)
    al = 5
    for j in range(N):
        for l in range(N):
            ten[j, 0][l, 0] += (np.exp(-np.abs(al) ** 2) * (al ** l) * (al ** j)
                                /(np.sqrt(float(np.math.factorial(j))) * np.sqrt(float(np.math.factorial(l)))))
    return ten/ten.reshape(3 * N, -1, order='F').trace()


def zerorho(n: int) -> np.array:
    """Returns a tensor of rank(4) with dimension (3N)^2."""
    ten = np.full((n, 3, n, 3), 0, dtype=complex)
    return ten


def helper(rho) -> np.array:
    """Helper function, which computes rho-dot, for a given density operator rho. Is used in Runge function,
    to iterate either with euler,
    Runge-Kutta method, in order to solve a first order differential equation at time t."""
    rho1 = zerorho(n=N)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    var = rhodot([j, m], [l, n], rho)
                    rho1[j, m][l, n] = var
    tester = rho1.reshape(3 * N, -1, order='F')
    assert np.matmul(tester, tester).all() == tester.all(), 'Failed computation'
    return rho1


def helper2(rho) -> np.array:
    """Helper function, which computes rho-dot, for a given density operator rho. Is used in Runge function,
    to iterate either with euler,
    Runge-Kutta method, in order to solve a first order differential equation at time t."""
    rho1 = zerorho(n=N)
    for index, val in np.ndenumerate(rho1):
        j = index[0]
        m = index[1]
        l = index[2]
        n = index[3]
        var = rhodot([j, m], [l, n], rho)
        rho1[j, m][l, n] = var
    tester = rho1.reshape(3 * N, -1, order='F')
    assert np.matmul(tester, tester).all() == tester.all(), 'Failed computation'
    return rho1


def euler2(rho, n):
    """Computes the Euler integration."""
    rhos = []
    rhos.append(rho)
    if KEY2 is False:
        for i in range(n):
            rho1 = rhos[-1] + helper2(rhos[-1]) * deltas
            rhos.append(rho1)
            tester = rho1.reshape(3 * N, - 1, order='F')
            print(f'Trace Iteration:{i}', round(tester.trace(), 5),
                  '\nImag', round(np.amin(tester.imag), 5), round(np.amax(tester.imag), 5),
                  'Real', round(np.amin(tester.real), 5), round(np.amax(tester.real), 5),
                  f'\nLen of rho: {len(rhos)}')
            if i % Iterstep == 0 and i > 2:
                step = i/Iterstep
                path = os.path.join(os.getcwd(), f'Euler{NAME}{str(itera)}_{N}_{deltas}_C{KEY2}_iter{int(step)}.npy')
                with open(path, 'wb') as file:
                    np.save(file, np.array(rhos[1:]))
                del rhos[0: - 1]
            if i == n - 1:
                step = int(n/Iterstep)
                path = os.path.join(os.getcwd(), f'Euler{NAME}{str(itera)}_{N}_{deltas}_C{KEY2}_iter{int(step)}.npy')
                with open(path, 'wb') as file:
                    np.save(file, np.array(rhos))
    else:
        for i in range(n):
            if i > 3:
                del rhos[-2]
            rho1 = rhos[-1] + helper(rhos[-1]) * deltas
            rhos.append(rho1)
            tester = rho1.reshape(3 * N, - 1, order='F')
            print(f'Trace Iteration:{i}', round(tester.trace(), 5),
                  '\nImag', round(np.amin(tester.imag), 5), round(np.amax(tester.imag), 5),
                  'Real', round(np.amin(tester.real), 5), round(np.amax(tester.real), 5))
        path = os.path.join(os.getcwd(), f'Euler{NAME}{str(itera)}_{N}_{deltas}_C{KEY2}.npy')
        with open(path, 'wb') as file:
            np.save(file, np.array(rhos))


def runge2(rho, n):
    """Computes the Runge-Kutta integration."""
    rhos = []
    rhos.append(rho)
    if KEY2 is False:
        for i in range(n):
            k1 = helper2(rhos[-1])
            k2 = helper2(rhos[-1] + deltas / 2 * k1)
            k3 = helper2(rhos[-1] + deltas / 2 * k2)
            k4 = helper2(rhos[-1] + deltas * k3)
            rho1 = rhos[-1] + (k1 + 2 * k2 + 2 * k3 + k4) * deltas / 6
            rhos.append(rho1)
            tester = rho1.reshape(3 * N, - 1, order='F')
            print(f'Trace Iteration:{i}', round(tester.trace(), 5),
                  '\nImag', round(np.amin(tester.imag), 5), round(np.amax(tester.imag), 5),
                  'Real', round(np.amin(tester.real), 5), round(np.amax(tester.real), 5),
                  f'\nLen of rho: {len(rhos)}')
            if i % Iterstep == 0 and i > 2:
                step = i/Iterstep
                path = os.path.join(os.getcwd(), f'Runge{NAME}{str(itera)}_{N}_{deltas}_C{KEY2}_iter{int(step)}.npy')
                with open(path, 'wb') as file:
                    np.save(file, np.array(rhos[1:]))
                del rhos[0: - 1]
            if i == n - 1:
                step = int(n / Iterstep)
                path = os.path.join(os.getcwd(), f'Runge{NAME}{str(itera)}_{N}_{deltas}_C{KEY2}_iter{int(step)}.npy')
                with open(path, 'wb') as file:
                    np.save(file, np.array(rhos))
    else:
        for i in range(n):
            if i > 3:
                del rhos[-2]
            k1 = helper(rhos[-1])
            k2 = helper(rhos[-1] + deltas / 2 * k1)
            k3 = helper(rhos[-1] + deltas / 2 * k2)
            k4 = helper(rhos[-1] + deltas * k3)
            rho1 = rhos[-1] + (k1 + 2 * k2 + 2 * k3 + k4) * deltas / 6
            rhos.append(rho1)
            tester = rho1.reshape(3 * N, - 1, order='F')
            print(f'Trace Iteration:{i}', round(tester.trace(), 5),
                '\nImag', round(np.amin(tester.imag), 5), round(np.amax(tester.imag), 5),
                'Real', round(np.amin(tester.real), 5), round(np.amax(tester.real), 5))
        path = os.path.join(os.getcwd(), f'Runge{NAME}{str(itera)}_{N}_{deltas}_C{KEY2}.npy')
        with open(path, 'wb') as file:
            np.save(file, np.array(rhos))


Rho0 = initialrho3(n=N)
"""Initiates the program"""
try:
    Method[KEY](Rho0, itera)
except Exception as E:
    raise E('Error in computing the time-evolution')
