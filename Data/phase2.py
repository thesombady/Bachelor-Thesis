import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import jv
#os.chdir('uniruns')
# os.chdir('..')
# os.chdir('Coherent2')
os.chdir('..')
os.chdir('coherenttest')

N = 50
R = 20
# Path = 'RungeAbove10000_100_0.01_CFalse_iter1.npy'
Path = 'RungeAbove2000_50_0.001_CFalse_iter1.npy'
Mean = 'MeanAboveCoherent.npy'


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
            # val.append(np.exp(-abs(a) ** 2 / 2) * a ** j / np.sqrt(float(np.math.factorial(j))) * data[j, m][j, m])
            # val1 = np.exp(-np.abs(a) ** 2) * a ** (j) / np.sqrt(float(np.math.factorial(j))) * data[j, m][j, m]
            val2 =   np.exp(-np.abs(a) ** 2) * a ** (2 * j) / (float(np.math.factorial(j))) * data[j, m][j, m]
            val.append(val2)
    val = np.array(val)
    return (sum(val.imag) + sum(val.real)) / np.pi


def mean(data):
    zeros = np.full([N, 3, N, 3], 0, dtype=complex)
    for j in range(N):
        for m in range(3):
            for l in range(N):
                for n in range(3):
                    try:
                        zeros[j, m][l, n] += np.sqrt(j) * data[j - l, m][l, n]
                    except:
                        zeros[j, m][l, n] += 0
    b = zeros.reshape(3 * N, -1, order='F').trace()
    return b.real + b.imag


def coherent2(a):
    al = 5
    val = []
    for j in range(N):
        for m in range(3):
            var1 = np.exp(-np.abs(a) ** 2 / 2) * a ** j / np.sqrt(float(np.math.factorial(j)))
            var2 = np.exp(-np.abs(al) ** 2 / 2) * al ** j / np.sqrt(float(np.math.factorial(j)))
            val.append(var1 * var2 * data[j, m][j, m])
    val = np.array(val)
    return (sum(val.imag) + sum(val.real)) / np.pi


def coherent3(a):
    a_0 = np.sqrt(mean(data))
    val = []
    for m in range(3):
        for j in range(N):
            val1 = np.exp(-np.abs(a) ** 2 / 2) * a ** j / np.sqrt(float(np.math.factorial(j)))
            val2 = np.exp(-np.abs(a_0) ** 2 / 2) * np.conjugate(a_0) ** j / np.sqrt(float(np.math.factorial(j)))
            var = val1 * val2 * data[j, m][j, m]
            val.append(var)
    return (sum(val).real + sum(val).imag)/np.pi


def coherent4(a):
    #a_0 = mean(data)
    # print(a_0)
    val = []
    for m in range(3):
        for j in range(N):
            val1 = np.conjugate(a) ** j / np.sqrt(float(np.math.factorial(j))) * (j + 1) * jv(j + 1, 2 * a)
            val.append(val1)
    return np.abs(sum(np.array(val))) ** 2 * np.exp(-np.abs(a) ** 2) / (np.pi * a_0 ** 2)


a = 5
xvec = np.linspace(-a, a, R, dtype=complex)
veccoherent = np.vectorize(coherent4)
X, Y = np.meshgrid(xvec, xvec)
"""
PATHS = ['RungeAbove10000_100_0.01_CFalse_iter10.npy', 'RungeLasing10000_100_0.01_CFalse_iter10.npy',
         'RungeBelow10000_100_0.01_CFalse_iter10.npy']
lists = [' ', '(b)', '(a)']

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
    # plt.show()
    plt.savefig(name2(path))

"""
data = parser(Path)[2]
a_0 = parser(Mean)[2]
print(a_0)
zval = veccoherent(np.sqrt(X ** 2 + Y ** 2))
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}
cax = ax.imshow(zval.real, cmap='jet', extent=[-a, a, -a, a],
             vmin=np.amin(zval.real), vmax=np.amax(zval.real), interpolation='bilinear')
# ax.annotate(text='(b)', xy=[-5, 1.05], xycoords=ax.get_xaxis_transform(), size=17)
ax.set_ylabel(r'$\mathfrak{Im}(\alpha)$', size=15, labelpad=2)
ax.set_xlabel(r'$\mathfrak{R}(\alpha)$', size=15)
ax.set_title(f'Q-function when operating\n{name(Path)} the masing threshold', size=17)
plt.yticks([-5, 0, 5], size=17)
plt.xticks([-5, 0, 5], size=17)
plt.show()
# plt.savefig('AboveQFunction2.pdf')