import numpy as np
import matplotlib.pyplot as plt
import os
N = 100
Size = 3 * N


def name(path):
    if "Above" in path:
        return 'above'
    elif "Below" in path:
        return 'below'
    elif "Lasing" in path:
        return 'at lasing'
    else:
        raise KeyError("Can't locate name")


def parser(path) -> np.array:
    global w_2
    """Parses the data, such that it's located in an .npy file."""
    with open(path, 'rb') as file:
        data1 = np.load(file)
    if 'Above' in path:
        w_2 = 150
    elif 'Lasing' in path:
        w_2 = 37.5
    elif 'Below' in path:
        w_2 = 34
    else:
        raise TypeError('Could not match the top-level')
    return data1


gamma_h = 1  # 10 ** (-8)  # might have to use something similar
hbar = 1
g = 5 * gamma_h
w_f = 30 * gamma_h
w_0 = 0
w_1 = w_f
w_2 = 150
w = [w_0, w_1, w_2]
R = 100

os.chdir('uniruns')
path = 'RungeAbove10000_100_0.01_CFalse_iter10.npy'
data = parser(path)[-1].reshape(Size, - 1, order='F')


def wigner(rho, xvec, yvec):
    g = np.sqrt(2)
    M = np.prod(rho.shape[0])
    X, Y = np.meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1j * Y)
    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(M)])
    Wlist[0] = np.exp(-2 * np.abs(A) ** 2) / np.pi
    W = np.real(rho[0, 0] * Wlist[0].real)
    for n in range(1, M):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wlist[n])

    for m in range(1, M):
        temp = np.copy(Wlist[m])
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1])/np.sqrt(m)
        W += np.real(rho[m, m] * Wlist[m])
        for n in range(m + 1, M):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp)/np.sqrt(n)
            temp = np.copy(Wlist[n])
            Wlist[n] = temp2
            W += 2 * np.real(rho[m, n] * Wlist[n])

    return 0.5 * W * g ** 2

xvec = np.linspace(-5, 5, R)
yvec = np.linspace(-5, 5, R)
val = wigner(data, xvec, yvec)

xx, yy = np.meshgrid(xvec, yvec)
plt.contourf(xx, yy, val, cmap='jet',
             vmin=np.amin(val), vmax=np.amax(val))
plt.ylabel(r'$\mathfrak{Im}(\alpha)$', rotation=0)
plt.xlabel(r'$\mathfrak{R}(\alpha)$')
plt.title(f'Q-function when operating {name(path)} the masing threshold')
# plt.show()
plt.colorbar()
# plt.savefig(f'Q-function{name(path)}.png', dpi='figure')
plt.show()
