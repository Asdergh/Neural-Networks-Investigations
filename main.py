import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math as mt
import random as rd




class DATA_Vis():

    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.figure = plt.figure()
        self.plane = self.figure.add_subplot(projection="3d")
    
    def load_data(self):
        self.grid_flag = False
        data = pd.read_json(self.file_name)
        if len(data.to_numpy()[0]) == 3:
            self.x_cores = np.asarray(data.iloc[:, 0])
            self.y_cores = np.asarray(data.iloc[:, 1])
            self.z_cores = np.asarray(data.iloc[:, 2])
        else:
            self.x_cores = np.asarray(data.iloc[:, 0:100])
            self.y_cores = np.asarray(data.iloc[:, 100:200])
            self.z_cores = np.asarray(data.iloc[:, 200:300])
            self.grid_flag = True
    
    def plot_graph_animaiton(self):
        print(self.y_cores)
        def graph_surface(time):
            self.plane.clear()
            self.plane.scatter(self.x_cores[:, 0:time], self.y_cores[:, 0:time], self.z_cores[:, 0:time], c=self.z_cores[:, 0:time], cmap="viridis", s=0.089)
            self.plane.plot_surface(self.x_cores[:, 0:time], self.y_cores[:, 0:time], self.z_cores[:,0:time], alpha=0.5, cmap="bone")
        
        def graph_trajectory(time):
            self.plane.clear()
            self.plane.plot(self.x_cores[0:time], self.y_cores[0:time], self.z_cores[0:time], alpha=0.5, color="blue", linestyle="--")
            self.plane.scatter(self.x_cores[time], self.y_cores[time], self.z_cores[time], alpha=0.2, color="red")
            self.plane.quiver(self.x_cores[time], self.y_cores[time], self.z_cores[time], self.x_cores[time + 1], 0, 0, color="red")
            self.plane.quiver(self.x_cores[time], self.y_cores[time], self.z_cores[time], 0, self.y_cores[time + 1], 0, color="blue")
            self.plane.quiver(self.x_cores[time], self.y_cores[time], self.z_cores[time], 0, 0, self.z_cores[time + 1], color="green")

        
        if self.grid_flag == True:
            animo = manimation.FuncAnimation(self.figure, graph_surface, interval=100)
            plt.show()
        else:
            animo = manimation.FuncAnimation(self.figure, graph_trajectory, interval=100)
            plt.show()


data = DATA_Vis(file_name="trajectory.json")
data.load_data()
data.plot_graph_animaiton()

def lorenc(xyz, s=10, r=28, b=2.668):
    x, y, z = xyz
    x_curent = s * (y - x)
    y_curent = r * x - y - x * z
    z_curent = x * y - b * z
    
    return np.array([x_curent, y_curent, z_curent])

dt = 0.01
xyz = np.empty((10000 + 1, 3))
xyz[0] = (0., 1., 1.05)
for iter in range(10000):
    xyz[iter + 1] = xyz[iter] + lorenc(xyz[iter]) * dt
#xyz = xyz.T

cores = pd.DataFrame(xyz)
print(cores.to_numpy().shape)
cores.to_json("trajectory.json")




        
        



