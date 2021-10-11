import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation
import matplotlib



def parser(path):
	"""A parser function, which utalizes the functionality of numpy.save. Using this method,
	each iteration can be imported, where then Data is an array of the different iterations."""
	with open(path, 'rb') as file:
		data = np.load(file)
	return data


"""
cax1 = plt.imshow(Data[0].reshape(Shape, - 1).real, extent=[0, Shape, 0, Shape], origin='lower',
				animated=False, interpolation='bilinear')  # , vmax = 1e-4, vmin = 0)
cax2 = plt.imshow(Data[0].reshape(Shape, - 1).imag, extent=[0, Shape, 0, Shape], origin='lower', 
	animated=False, interpolation='bilinear')
"""

"""
def animate(n):
	dataset = Data[n].reshape(Shape, - 1)
	cax1.set_array(dataset.real)
	plt.title(f'Iteration {n}')
	return fig,


ani = FuncAnimation(fig, animate1, interval=1, frames=len(Data))
#plt.colorbar()

"""

Path = os.path.join(os.getcwd(), 'EulerAbove1000_100.npy')
Data = parser(Path)
Shape = 3 * 100


fig = plt.figure()
FPS = 30
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
iterations = []
for i in range(len(Data)):
	plt.title(f'Iteration {i}')
	im1 = ax1.imshow(Data[i].reshape(Shape, - 1).real, extent=[0, Shape, 0, Shape],
					origin='lower', animated=False, interpolation='bilinear', vmax=1, vmin=-1)
	im2 = ax2.imshow(Data[i].reshape(Shape, - 1).imag, extent=[0, Shape, 0, Shape],
					origin='lower', animated=False, interpolation='bilinear', vmax=1, vmin=-1)
	fig.colorbar(im1, cax=ax1)
	fig.colorbar(im2, cax=ax2)
	iterations.append([im1, im2])

ani = ArtistAnimation(fig, iterations, blit=True, interval=50)
name = 'EulerAbove1000_100.mp4'
path = os.path.join(os.path.join(os.getcwd(), 'Sim', name))
print("ani made")
#writer = AnimatedPNGWriter(fps=FPS)
matplotlib.rcParams['animation.FFmpeg_path'] = r'/Users/andreasevensen/opt/anaconda3/pkgs/ffmpeg-4.3.2-h4dad6da_0/bin/ffmpeg '
"""
try:
	print("first")
	ani.save(path, fps=FPS, writer='pillow')
except Exception as e:
	raise e
"""
print("about to ")
writer = matplotlib.animation.FFMpegWriter()
ani.save(path, fps=FPS, writer=f'{writer}')
plt.show()

"""
vid = ani.to_html5_video()
html = display.HTML(vid)
display.display(html)
plt.close()
print(vid)
"""