import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

figure = plt.figure()
surface = figure.add_subplot(projection="3d")
plt.style.use("seaborn")

def animation(time):
    surface.clear()
    xx1_grid, xx2_grid = np.meshgrid(np.linspace(-np.pi, np.pi, 100),
                                     np.linspace(-np.pi, np.pi, 100))
    xx3_grid = np.sin(np.sqrt(xx1_grid ** 2 * time + xx2_grid ** 2 * time) + np.exp(xx1_grid ** 2 * time + xx2_grid ** 3 * time)) / (np.sqrt(xx1_grid ** 2 * time + xx2_grid ** 2 * time))
    surface.plot_surface(xx1_grid, xx2_grid, xx3_grid, alpha=0.5, cmap="coolwarm")
    surface.scatter(xx1_grid, xx2_grid, xx3_grid, color="black", s=0.12)

animo = manimation.FuncAnimation(figure, animation, interval=100)
plt.show()


