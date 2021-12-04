import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
os.chdir('Coherent')

N = 100
R = 50
Path = 'RungeAbove10000_100_0.01_CFalse_iter1.npy'
FPS = 50


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


Dataset = parser(Path)


def coherent(a, t):
    data = Dataset[t]
    val = []
    for l in range(0, N):
        for m in range(3):
            # val += np.exp(-abs(a) ** 2 / 2) * a ** l / (np.sqrt(np.math.factorial(l)))
            val.append(np.exp(-abs(a) ** 2 / 2) * a ** l / np.sqrt(float(np.math.factorial(l))) * data[l, m][l, m])
    return sum(val).real/np.pi



a = 5
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

ani = FuncAnimation(fig, animate, interval=1, frames=int(len(Dataset)))
try:
    ani.save('test.gif', fps=FPS, writer='pillow', extra_args=['-vcodec', 'libx264'])
except Exception as E:
    ani.save('test.gif', fps=FPS, writer='pillow')
