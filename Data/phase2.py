import numpy as np
import matplotlib.pyplot as plt
import os
# os.chdir('uniruns')
os.chdir('..')
os.chdir('coherenttest')

N = 5  # 100
R = 50
# Path = 'RungeAbove10000_100_0.01_CFalse_iter1.npy'
Path = 'RungeAbove1000_5_0.01_CFalse_iter2.npy'


def parser(path):
    with open(path, 'rb') as file:
        return np.load(file)


def name(path) -> str:
    if "Above" in path:
        return 'above'
    elif "Below" in path:
        return 'below'
    elif "Lasing" in path:
        return 'at lasing'
    else:
        raise KeyError("Can't locate name")


def name2(path):
    if 'Above' in path:
        return 'AboveQFunction.pdf'
    elif 'Lasing' in path:
        return 'LasingQFunction.pdf'
    elif 'Below' in path:
        return 'BelowQFunction.pdf'
    else:
        raise TypeError('Can not save')


def coherent(a):
    val = []
    for j in range(N):
        for m in range(3):
            # val += np.exp(-abs(a) ** 2 / 2) * a ** l / (np.sqrt(np.math.factorial(l)))
            val.append(np.exp(-abs(a) ** 2 / 2) * a ** j / np.sqrt(float(np.math.factorial(j))) * data[j, m][j, m])
    return sum(val).real/np.pi


def coherent2(a):
    al = 1.25
    val = []
    for j in range(N):
        for m in range(3):
            var = np.exp(-abs(a) ** 2 / 2 - abs(al) ** 2 / 2) * a ** j / np.sqrt(float(np.math.factorial(j))) * data[j, m][j, m]


a = 5
xvec = np.linspace(-a, a, R, dtype=complex)
veccoherent = np.vectorize(coherent)
X, Y = np.meshgrid(xvec, xvec)
PATHS = ['RungeAbove10000_100_0.01_CFalse_iter10.npy', 'RungeLasing10000_100_0.01_CFalse_iter10.npy',
         'RungeBelow10000_100_0.01_CFalse_iter10.npy']
lists = ['(c)', '(b)', '(a)']
"""
for path, const in zip(PATHS, lists):
    global data
    data = parser(path)[-1]

    zval = veccoherent(np.sqrt(X ** 2 + Y ** 2))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    font = {'family': 'normal',
            'size': 13}
    cax = ax.imshow(zval, cmap='jet', extent=[-a, a, -a, a],
                 vmin=np.amin(zval), vmax=np.amax(zval))
    ax.annotate(text=const, xy=[-5, 1.05], xycoords=ax.get_xaxis_transform(), size=17)
    ax.set_ylabel(r'$\mathfrak{Im}(\alpha)$', size=15, labelpad=2)
    ax.set_xlabel(r'$\mathfrak{R}(\alpha)$', size=15)
    ax.set_title(f'Q-function when operating\n{name(path)} the masing threshold', size=17)
    plt.yticks([-5, 0, 5], size=17)
    plt.xticks([-5, 0, 5], size=17)
    plt.show()
    # plt.savefig(name2(path))
"""
data = parser(Path)[-1]
zval = veccoherent(np.sqrt(X ** 2 + Y ** 2))
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}
cax = ax.imshow(zval, cmap='jet', extent=[-a, a, -a, a],
             vmin=np.amin(zval), vmax=np.amax(zval))
ax.annotate(text='test', xy=[-5, 1.05], xycoords=ax.get_xaxis_transform(), size=17)
ax.set_ylabel(r'$\mathfrak{Im}(\alpha)$', size=15, labelpad=2)
ax.set_xlabel(r'$\mathfrak{R}(\alpha)$', size=15)
ax.set_title(f'Q-function when operating\n{name(Path)} the masing threshold', size=17)
plt.yticks([-5, 0, 5], size=17)
plt.xticks([-5, 0, 5], size=17)
plt.show()