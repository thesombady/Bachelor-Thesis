import numpy as np
import matplotlib.pyplot as plt

alpha_0 = 3.98
delta = 0.01
hbar = 1
N = 5


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def func(t, w):
    return alpha_0 * np.exp(-1j * w * t)


def derivative(func, t, w):
    h = 1e-9
    return (1/h) * (func(t + h, w) - func(t, w))


def plotmaker():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    t = np.linspace(0, 100, 10000)
    phi = 10
    delx = hbar/4
    delp = hbar/4
    val = func(t, phi);
    x = val.real
    p = val.imag
    grid = np.meshgrid(x, p)
    # plt.imshow(grid[0], extent=[0, 100, 0, 100])
    ax.plot(x, p, color='blue')
    ax.set_xlabel(r'$\mathcal{R}(\alpha)$')
    ax.set_ylabel(r'$\mathcal{I}(\alpha)$')
    ax.set_title('Phase diagram')
    #print(grid[-1])
    plt.show()


def plotish(data):
    plt.hist(data, N, log=True)
    plt.show()


def plotish2(data):
    xlist = np.linspace(0, N, 1)



def plotmaker2(rhos):
    occupation = np.full((N, 3, N, 3), 0, dtype=complex)
    number = []
    for rho in rhos:
        for j in range(N):
            for m in range(3):
                for l in range(N):
                    for n in range(3):
                        occupation[j, m][l, n] = j * rho[j, m][l, n]
        test = occupation.reshape(3 * N, - 1, order='F')
        number.append(test.trace())
    plotish(number)
    # print(number)






Path = 'RungeAbove10000_5_0.01.npy'
data = parser(Path)
# plotmaker2(data)
xlist = np.linspace(0, 1, 1e9)
ylist =






