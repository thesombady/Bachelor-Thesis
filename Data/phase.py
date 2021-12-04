import numpy as np
import matplotlib.pyplot as plt
import os
N = 100
Size = 3 * N
R = 50


def name(path) -> str:
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


def name2(path):
    if 'Above' in path:
        return 'AboveQFunction.pdf'
    elif 'Lasing' in path:
        return 'LasingQFunction.pdf'
    elif 'Below' in path:
        return 'BelowQFunction.pdf'
    else:
        raise TypeError('Can not save')


gamma_h = 1  # 10 ** (-8)  # might have to use something similar
hbar = 1
g = 5 * gamma_h
w_f = 30 * gamma_h
w_0 = 0
w_1 = w_f
w_2 = 150
w = [w_0, w_1, w_2]


os.chdir('uniruns')
path = 'RungeBelow10000_100_0.01_CFalse_iter10.npy'
data = parser(path)[-1].reshape(Size, - 1) #  , order='F')


def wigner(rho, xvec, yvec):
    g = 2
    M = np.prod(rho.shape[0])
    X, Y = np.meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)
    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(M)])
    Wlist[0] = np.exp(-2.0 * abs(A) ** 2) / np.pi

    W = np.real(rho[0, 0]) * np.real(Wlist[0])

    for n in range(1, M):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2.0 * np.real(rho[0, n] * Wlist[n])

    for m in range(1, M):
        temp = np.copy(Wlist[m])
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1]) / np.sqrt(m)

        W += np.real(rho[m, m] * Wlist[m])
        for n in range(m + 1, M):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = np.copy(Wlist[n])
            Wlist[n] = temp2

            W += 2 * np.real(rho[m, n] * Wlist[n])
    return 1/2 * W * g ** 2


xvec = np.linspace(-5, 5, R)
yvec = np.linspace(-5, 5, R)
val = wigner(data, xvec, yvec)

xx, yy = np.meshgrid(xvec, yvec)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}
cax = ax.imshow(val, cmap='jet', extent=[-5, 5, -5, 5],
             vmin=np.amin(val), vmax=np.amax(val))
ax.set_ylabel(r'$\mathfrak{Im}(\alpha)$', size=15, labelpad=2)
ax.set_xlabel(r'$\mathfrak{R}(\alpha)$', size=15)
ax.set_title(f'Q-function when operating\n{name(path)} the masing threshold', size=17)
plt.yticks([-5, 0, 5])
plt.xticks([-5, 0, 5])
# plt.show()
plt.savefig(name2(path))
