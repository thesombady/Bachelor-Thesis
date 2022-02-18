import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
# os.chdir('Coherent2')
os.chdir('..')
os.chdir('coherenttest')

N = 50
R = 50
Path1 = 'RungeAbove2000_50_0.001_CFalse_iter1.npy'
Path2 = 'MeanAboveCoherent.npy'
FPS = 30


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


Dataset = parser(Path1)
oc = parser(Path2)


def coherent(a, t):
    data = Dataset[t]
    a_0 = np.sqrt(oc[t])
    val = []
    for m in range(3):
        for j in range(N):
            for l in range(N):
                val1 = np.exp(-np.abs(a) ** 2 / 2) * a ** j / np.sqrt(float(np.math.factorial(j)))
                val2 = np.exp(-np.abs(a_0) ** 2 / 2) * np.conjugate(a_0) ** l / np.sqrt(float(np.math.factorial(l)))
                var = val1 * val2 * data[l, m][j, m]
                val.append(var)
    return (sum(val).real + sum(val).imag) / np.pi


def coherent2(a, t):
    data = Dataset[t]
    a_0 = oc[t]
    a_1 = 5
    val = []
    for m in range(3):
        for j in range(N):
            val1 = np.exp(-np.abs(a_0) ** 2 / 2) * a_0 ** 2 / np.sqrt(float(np.math.factorial(j)))
            val2 = np.exp(-np.abs(a_1) ** 2 / 2) * a_1 ** 2 / np.sqrt(float(np.math.factorial(j)))
            val.append(val1 * val2 * data[j, m][j, m])
    return (sum(val).real + sum(val).real) / np.pi




a = 7
xvec = np.linspace(-a, a, R)
X, Y = np.meshgrid(xvec, xvec)

veccoherent = np.vectorize(coherent)
mesh = np.sqrt(X ** 2 + Y ** 2)
zval = veccoherent(mesh, 0)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family': 'normal',
        'size': 13}
cax = ax.imshow(zval, cmap='jet', extent=[-a, a, -a, a],
             vmin=np.amin(zval), vmax=np.amax(zval))
ax.annotate(text='(a)', xy=[-5, 1.05], xycoords=ax.get_xaxis_transform(), size=17)
ax.set_ylabel(r'$\mathfrak{Im}(\alpha)$', size=15, labelpad=2)
ax.set_xlabel(r'$\mathfrak{R}(\alpha)$', size=15)
# ax.set_title(f'Q-function when operating\n{name(Path)} the masing threshold', size=17)
plt.yticks([-5, 0, 5], size=17)
plt.xticks([-5, 0, 5], size=17)
# plt.show()
# plt.savefig(name2(Path))


def animate(t):
    val = Dataset[t]
    cax.set_array(veccoherent(mesh, t))
    fig.suptitle(f'Time {0.01 * t}')
    return fig,

ani = FuncAnimation(fig, animate, interval=1, frames=60)
try:
    ani.save('test.gif', fps=FPS, writer='pillow', extra_args=['-vcodec', 'libx264'])
except Exception as E:
    ani.save('test.gif', fps=FPS, writer='pillow')
