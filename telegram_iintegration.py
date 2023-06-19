import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn")
surface = plt.figure().add_subplot(projection="3d")
x_grid, y_grid = np.meshgrid(np.linspace(-np.pi, np.pi, 100), 
                             np.linspace(-np.pi, np.pi, 100))

z_grid = y_grid - x_grid - 5
z_grid_2 = (-7 * x_grid - 9 * y_grid + 17) / 2
z_grid_3 = (12 * x_grid - 5 * y_grid) / 4
surface.plot_surface(x_grid, y_grid, z_grid, color="green", alpha=0.3)
surface.plot_surface(x_grid, y_grid, z_grid_2, color="red", alpha=0.3)
surface.plot_surface(x_grid, y_grid, z_grid_3, color="blue", alpha=0.3)
plt.show()