import warnings
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


warnings.simplefilter('ignore')

def func_z(x, y):
    # Calculate values of Z from the created grid
    z = x**2/5. + x*y/50. + y**2/5.

    return z


def gradient_descent(previous_x, previous_y, learning_rate, epoch):
    x_gd = []
    y_gd = []
    z_gd = []

    x_gd.append(previous_x)
    y_gd.append(previous_y)
    z_gd.append(func_z(previous_x, previous_y))

    # begin the loops to update x, y and z
    for i in range(epoch):
        current_x = previous_x - learning_rate*(2*previous_x/5. +
                                               previous_y/50.)
        x_gd.append(current_x)
        current_y = previous_y - learning_rate*(previous_x/50. +
                                                previous_y/5.)
        y_gd.append(current_y)

        z_gd.append(func_z(current_x, current_y))

        # update previous_x and previous_y
        previous_x = current_x
        previous_y = current_y

    return x_gd, y_gd, z_gd


x0 = -2
y0 = 2.5
learning_rate = 1.3
epoch = 10


# plot function
a = np.arange(-3, 3, 0.05)
b = np.arange(-3, 3, 0.05)

x, y = np.meshgrid(a, b)
z = func_z(x, y)

fig1 = plt.figure()
ax1 = Axes3D(fig1)
surf = ax1.plot_surface(x, y, z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')

# Plot target (the minimum of the function)
min_point = np.array([0., 0.])
min_point_ = min_point[:, np.newaxis]
ax1.plot(*min_point_, func_z(*min_point_), 'r*', markersize=10)

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$z$')

#run gradient descent
x_gd, y_gd, z_gd = gradient_descent(x0, y0, learning_rate, epoch)

# Create animation
line, = ax1.plot([], [], [], 'r-', label='Gradient descent', lw=1.5)
point, = ax1.plot([], [], [], 'bo')
display_value = ax1.text(2., 2., 27.5, '', transform=ax1.transAxes)


def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    display_value.set_text('')

    return line, point, display_value


def animate(i):
    # Animate line
    line.set_data(x_gd[:i], y_gd[:i])
    line.set_3d_properties(z_gd[:i])

    # Animate points
    point.set_data(x_gd[i], y_gd[i])
    point.set_3d_properties(z_gd[i])

    # Animate display value
    display_value.set_text('Min = ' + str(z_gd[i]))

    return line, point, display_value


ax1.legend(loc=1)

anim = animation.FuncAnimation(fig1, animate, init_func=init,
                               frames=len(x_gd), interval=120,
                               repeat_delay=90, blit=True)


plt.show()
